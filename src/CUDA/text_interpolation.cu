#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <limits>
#include <algorithm>
#include "../Host_Device_Shared/helpers.h"
#include "../Host_Device_Shared/Interpolation.h"
#include "color.cuh"

// Various functions used for text interpolation, which takes two rasterized LaTeX images,
// assumed to be mathematical formulas, and interpolates between them by identifying shared glyphs.
// The algorithm is as follows:
// 1.   Convert the grayscale images to black and white with a simple brightness threshold.
// 2.   Identify and enumerate all connected components in each image by naively setting the colors
//      to the uint32_t index of the component. This is done with a simple flood fill algorithm.
// 3.   Identify which pairs of components are the same glyph.
// 3a.  Immediately discard pairs whose bounding boxes have aspect ratios that differ by more than a few percent
//      (measured as |log(ar1 / ar2)|, so the tolerance is relative rather than absolute).
// 3b.  For the remaining pairs, compute IOU (Intersection over Union) of the glyphs, and discard pairs with IOU far from 1.
//      We do a best-effort scale to make the bounding boxes of the components match before computing IOU,
//      to account for differently sized glyphs, such as exponents turning into coefficients in a derivative.
// 4.   We are now left with a bipartite graph of components, where edges connect shape-compatible glyphs.
//      Choose the one-to-one matching with the least total on-screen travel distance, solved using the
//      Hungarian algorithm. Unmatched leftover glyphs fade out or in.
// 5.   We now have paired shared glyphs. We can interpolate between them by linearly interpolating
//      the positions of the glyphs, and using a simple alpha blend for the pixels in the glyphs.

const float ASPECT_RATIO_THRESHOLD = 0.25f;
const float IOU_THRESHOLD = 0.75f;

// Helper function to get the bounding box of a glyph with a given color
// NOTE: Due to max operation on matching pixels, d_bottom_right is inclusive (not exclusive)
//       Add 1 to d_bottom_right after calling if needed
__global__ void get_bounding_box_kernel(const uint32_t* pix, const Cuda::ivec2 wh, uint32_t color, Cuda::ivec2* d_top_left, Cuda::ivec2* d_bottom_right)
{
    Cuda::ivec2 grid_point = Cuda::ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    if (grid_point.x >= wh.x || grid_point.y >= wh.y) return;
    if (pix[grid_point.y * wh.x + grid_point.x] == color) {
        atomicMin(&d_top_left->x, grid_point.x);
        atomicMin(&d_top_left->y, grid_point.y);
        atomicMax(&d_bottom_right->x, grid_point.x);
        atomicMax(&d_bottom_right->y, grid_point.y);
    }
}

__global__ void copy_glyph_to_gpu_kernel(const uint32_t* pix, const Cuda::ivec2 wh, const Cuda::ivec2 top_left, const Cuda::ivec2 bottom_right, uint32_t* d_glyph_pix, const uint32_t color)
{
    Cuda::ivec2 grid_point = Cuda::ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const Cuda::ivec2 glyph_wh = bottom_right - top_left;
    if (grid_point.x >= glyph_wh.x || grid_point.y >= glyph_wh.y) return;
    Cuda::ivec2 pix_point = top_left + grid_point;

    uint32_t glyph_color = pix[pix_point.y * wh.x + pix_point.x] == color ? 0xFFFFFFFF : 0x00FFFFFF;
    d_glyph_pix[grid_point.y * glyph_wh.x + grid_point.x] = glyph_color;
}

__global__ void compute_iou_kernel (
    const uint32_t* pix_1, const Cuda::ivec2 wh_1,
    const uint32_t* pix_2, const Cuda::ivec2 wh_2,
    uint32_t* intersection, uint32_t* union_count)
{
    Cuda::ivec2 grid_point = Cuda::ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    if (grid_point.x >= 100 || grid_point.y >= 100) return;

    Cuda::vec2 norm = (Cuda::vec2(grid_point.x + 0.5f, grid_point.y + 0.5f)) / 100.0f;

    Cuda::ivec2 p1(norm.x * wh_1.x, norm.y * wh_1.y);
    Cuda::ivec2 p2(norm.x * wh_2.x, norm.y * wh_2.y);

    bool a = pix_1[p1.y * wh_1.x + p1.x] == 0xFFFFFFFF;
    bool b = pix_2[p2.y * wh_2.x + p2.x] == 0xFFFFFFFF;

    if (a || b) atomicAdd(union_count, 1u);
    if (a && b) atomicAdd(intersection, 1u);
}

float compute_iou (
    const uint32_t* pix_1, const Cuda::ivec2 wh_1,
    const uint32_t* pix_2, const Cuda::ivec2 wh_2)
{
    uint32_t* d_intersection;
    uint32_t* d_union;
    uint32_t h_intersection = 0;
    uint32_t h_union = 0;
    cudaMalloc(&d_intersection, sizeof(uint32_t));
    cudaMalloc(&d_union, sizeof(uint32_t));
    cudaMemcpy(d_intersection, &h_intersection, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_union, &h_union, sizeof(uint32_t), cudaMemcpyHostToDevice);
    // TODO do we really need to do a memcpy for something this simple??

    dim3 threads(10, 10);
    dim3 blocks(10, 10);
    compute_iou_kernel<<<blocks, threads>>>(pix_1, wh_1, pix_2, wh_2, d_intersection, d_union);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_intersection, d_intersection, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_union, d_union, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_intersection);
    cudaFree(d_union);

    if (h_union == 0) return 0.0f;
    return (float) h_intersection / (float) h_union;
}

bool is_same_glyph(const Cuda::Glyph& g1, const Cuda::Glyph& g2)
{
    // Degenerate (empty) bounding boxes can never match.
    if (g1.wh.x <= 0 || g1.wh.y <= 0 || g2.wh.x <= 0 || g2.wh.y <= 0) {
        return false;
    }
    float aspect_ratio_1 = (float) g1.wh.x / (float) g1.wh.y;
    float aspect_ratio_2 = (float) g2.wh.x / (float) g2.wh.y;
    if (std::abs(std::log(aspect_ratio_1 / aspect_ratio_2)) > ASPECT_RATIO_THRESHOLD) {
        return false;
    }
    float iou = compute_iou(g1.pix, g1.wh, g2.pix, g2.wh);
    if (iou < IOU_THRESHOLD) {
        return false;
    }
    return true;
}

void print_matching(const Cuda::Interpolation& interpolation)
{
    for (int i = 0; i < interpolation.num_glyphs_1; i++) {
        if (interpolation.match_1[i] >= 0) printf("%d -> %d\n", i, interpolation.match_1[i]);
        else printf("%d fades out\n", i);
    }
    for (int j = 0; j < interpolation.num_glyphs_2; j++) {
        if (interpolation.match_2[j] < 0) printf("%d fades in\n", j);
    }
}

// On-screen distance glyph i (image 1) travels to reach glyph j (image 2), in the centered layout
// interpolate() draws with (each image centered in the output), so differing image sizes don't bias it.
float travel_distance(const Cuda::Interpolation& interpolation, uint32_t i, uint32_t j)
{
    const Cuda::Glyph& g1 = interpolation.glyphs_1[i];
    const Cuda::Glyph& g2 = interpolation.glyphs_2[j];
    const float x1 = g1.top_left.x + 0.5f * g1.wh.x - 0.5f * interpolation.wh_1.x;
    const float y1 = g1.top_left.y + 0.5f * g1.wh.y - 0.5f * interpolation.wh_1.y;
    const float x2 = g2.top_left.x + 0.5f * g2.wh.x - 0.5f * interpolation.wh_2.x;
    const float y2 = g2.top_left.y + 0.5f * g2.wh.y - 0.5f * interpolation.wh_2.y;
    return std::hypot(x2 - x1, y2 - y1);
}

// Minimum-cost assignment on a square n x n cost matrix (row-major). Returns assignment[row] = column.
// O(n^3) Hungarian algorithm with potentials (Kuhn-Munkres, shortest augmenting path formulation).
std::vector<int> solve_assignment(const std::vector<double>& cost, const int n)
{
    const double INF = std::numeric_limits<double>::infinity();
    // 1-indexed internally; column 0 is a sentinel.
    std::vector<double> u(n + 1, 0), v(n + 1, 0);
    std::vector<int> p(n + 1, 0), way(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, INF);
        std::vector<char> used(n + 1, 0);
        do {
            used[j0] = 1;
            const int i0 = p[j0];
            double delta = INF;
            int j1 = 0;
            for (int j = 1; j <= n; j++) {
                if (used[j]) continue;
                const double cur = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= n; j++) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    std::vector<int> assignment(n, -1);
    for (int j = 1; j <= n; j++) if (p[j] > 0) assignment[p[j] - 1] = j - 1;
    return assignment;
}

// Steps 3 and 4: among the pairs that pass the shape tests (is_same_glyph), choose the matching with the most
// matched pairs, breaking ties by least total travel distance. Writes the result to match_1 and match_2.
// Solved as a square n x n assignment problem with n = max(n1, n2): rows are image 1 glyphs, columns are image 2
// glyphs, and the shorter side is padded with free dummies.
void solve_matching(Cuda::Interpolation& interpolation)
{
    const int n1 = interpolation.num_glyphs_1, n2 = interpolation.num_glyphs_2;
    const int n = std::max(n1, n2);
    for (int i = 0; i < n1; i++) interpolation.match_1[i] = -1;
    for (int j = 0; j < n2; j++) interpolation.match_2[j] = -1;
    if (n1 == 0 || n2 == 0) return;

    const double diag = std::max(std::hypot((double) interpolation.wh_1.x, (double) interpolation.wh_1.y),
                                 std::hypot((double) interpolation.wh_2.x, (double) interpolation.wh_2.y));
    const double forbidden = diag * (std::min(n1, n2) + 1);

    std::vector<double> cost((size_t) n * n, 0); // Dummy rows/columns (beyond n1 or n2) stay free
    for (int i = 0; i < n1; i++)
        for (int j = 0; j < n2; j++)
            cost[(size_t) i * n + j] = is_same_glyph(interpolation.glyphs_1[i], interpolation.glyphs_2[j])
                                       ? travel_distance(interpolation, i, j) : forbidden;

    const std::vector<int> assignment = solve_assignment(cost, n);

    for (int i = 0; i < n1; i++) {
        const int j = assignment[i];
        if (j < n2 && cost[(size_t) i * n + j] < forbidden) {
            interpolation.match_1[i] = j;
            interpolation.match_2[j] = i;
        }
    }
}

// The main entry point which stages an interpolation.
// The CPU is expected to have done steps 1 and 2 already, since flood filling is simpler on the CPU.
extern "C" Cuda::Interpolation stage_interpolation(
    const uint32_t* h_pix_1, const Cuda::ivec2 wh_1, const int num_glyphs_1,
    const uint32_t* h_pix_2, const Cuda::ivec2 wh_2, const int num_glyphs_2)
{
    Cuda::Interpolation interpolation;

    interpolation.wh_1 = wh_1;
    interpolation.wh_2 = wh_2;

    cudaMalloc(&interpolation.pix_1, wh_1.x * wh_1.y * sizeof(uint32_t));
    cudaMalloc(&interpolation.pix_2, wh_2.x * wh_2.y * sizeof(uint32_t));
    cudaMemcpy(interpolation.pix_1, h_pix_1, wh_1.x * wh_1.y * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(interpolation.pix_2, h_pix_2, wh_2.x * wh_2.y * sizeof(uint32_t), cudaMemcpyHostToDevice);

    interpolation.num_glyphs_1 = num_glyphs_1;
    interpolation.num_glyphs_2 = num_glyphs_2;
    interpolation.glyphs_1 = new Cuda::Glyph[num_glyphs_1];
    interpolation.glyphs_2 = new Cuda::Glyph[num_glyphs_2];
    interpolation.match_1 = new int[num_glyphs_1];
    interpolation.match_2 = new int[num_glyphs_2];

    // Allocate two ivec2s on the GPU to hold the top left and bottom right corners of the bounding box
    Cuda::ivec2* d_top_left;
    Cuda::ivec2* d_bottom_right;
    cudaMalloc(&d_top_left, sizeof(Cuda::ivec2));
    cudaMalloc(&d_bottom_right, sizeof(Cuda::ivec2));

    for (int i = 0; i < num_glyphs_1; i++) {
        // Populate the ivec2s
        Cuda::ivec2 top_left(wh_1.x, wh_1.y);
        Cuda::ivec2 bottom_right(0, 0);
        cudaMemcpy(d_top_left, &top_left, sizeof(Cuda::ivec2), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bottom_right, &bottom_right, sizeof(Cuda::ivec2), cudaMemcpyHostToDevice);
        dim3 threads(16, 16);
        dim3 blocks((wh_1.x + threads.x - 1) / threads.x, (wh_1.y + threads.y - 1) / threads.y);
        get_bounding_box_kernel<<<blocks, threads>>>(interpolation.pix_1, wh_1, i + 1, d_top_left, d_bottom_right);
        cudaDeviceSynchronize();
        cudaMemcpy(&top_left, d_top_left, sizeof(Cuda::ivec2), cudaMemcpyDeviceToHost);
        cudaMemcpy(&bottom_right, d_bottom_right, sizeof(Cuda::ivec2), cudaMemcpyDeviceToHost);
        bottom_right += Cuda::ivec2(1, 1);
        interpolation.glyphs_1[i].top_left = top_left;
        interpolation.glyphs_1[i].wh = bottom_right - top_left;
        cudaMalloc(&interpolation.glyphs_1[i].pix, interpolation.glyphs_1[i].wh.x * interpolation.glyphs_1[i].wh.y * sizeof(uint32_t));
        threads = dim3(16, 16);
        blocks = dim3((interpolation.glyphs_1[i].wh.x + threads.x - 1) / threads.x, (interpolation.glyphs_1[i].wh.y + threads.y - 1) / threads.y);
        copy_glyph_to_gpu_kernel<<<blocks, threads>>>(interpolation.pix_1, wh_1, top_left, bottom_right, interpolation.glyphs_1[i].pix, i + 1);
    }

    for (int i = 0; i < num_glyphs_2; i++) {
        Cuda::ivec2 top_left(wh_2.x, wh_2.y);
        Cuda::ivec2 bottom_right(0, 0);
        cudaMemcpy(d_top_left, &top_left, sizeof(Cuda::ivec2), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bottom_right, &bottom_right, sizeof(Cuda::ivec2), cudaMemcpyHostToDevice);
        dim3 threads(16, 16);
        dim3 blocks((wh_2.x + threads.x - 1) / threads.x, (wh_2.y + threads.y - 1) / threads.y);
        get_bounding_box_kernel<<<blocks, threads>>>(interpolation.pix_2, wh_2, i + 1, d_top_left, d_bottom_right);
        cudaDeviceSynchronize();
        cudaMemcpy(&top_left, d_top_left, sizeof(Cuda::ivec2), cudaMemcpyDeviceToHost);
        cudaMemcpy(&bottom_right, d_bottom_right, sizeof(Cuda::ivec2), cudaMemcpyDeviceToHost);
        bottom_right += Cuda::ivec2(1, 1);
        interpolation.glyphs_2[i].top_left = top_left;
        interpolation.glyphs_2[i].wh = bottom_right - top_left;
        cudaMalloc(&interpolation.glyphs_2[i].pix, interpolation.glyphs_2[i].wh.x * interpolation.glyphs_2[i].wh.y * sizeof(uint32_t));
        threads = dim3(16, 16);
        blocks = dim3((interpolation.glyphs_2[i].wh.x + threads.x - 1) / threads.x, (interpolation.glyphs_2[i].wh.y + threads.y - 1) / threads.y);
        copy_glyph_to_gpu_kernel<<<blocks, threads>>>(interpolation.pix_2, wh_2, top_left, bottom_right, interpolation.glyphs_2[i].pix, i + 1);
    }

    cudaFree(d_top_left);
    cudaFree(d_bottom_right);

    solve_matching(interpolation);
    print_matching(interpolation);

    return interpolation;
}

extern "C" void free_interpolation(Cuda::Interpolation& interpolation)
{
    cudaFree(interpolation.pix_1);
    cudaFree(interpolation.pix_2);
    for (int i = 0; i < interpolation.num_glyphs_1; i++) {
        cudaFree(interpolation.glyphs_1[i].pix);
    }
    for (int i = 0; i < interpolation.num_glyphs_2; i++) {
        cudaFree(interpolation.glyphs_2[i].pix);
    }
    delete[] interpolation.glyphs_1;
    delete[] interpolation.glyphs_2;
    delete[] interpolation.match_1;
    delete[] interpolation.match_2;
}







// ------------------
// Runtime interpolation
// ------------------

__device__ uint32_t sample_bilinear(const Cuda::Glyph& glyph, const Cuda::vec2 uv)
{
    float x = uv.x * (glyph.wh.x - 1);
    float y = uv.y * (glyph.wh.y - 1);

    int x0 = max(0, min(glyph.wh.x - 1, (int) floorf(x)));
    int y0 = max(0, min(glyph.wh.y - 1, (int) floorf(y)));
    int x1 = max(0, min(glyph.wh.x - 1, x0 + 1));
    int y1 = max(0, min(glyph.wh.y - 1, y0 + 1));

    float tx = x - x0;
    float ty = y - y0;

    uint32_t c00 = glyph.pix[y0 * glyph.wh.x + x0];
    uint32_t c10 = glyph.pix[y0 * glyph.wh.x + x1];
    uint32_t c01 = glyph.pix[y1 * glyph.wh.x + x0];
    uint32_t c11 = glyph.pix[y1 * glyph.wh.x + x1];

    return Cuda::colorlerp(Cuda::colorlerp(c00, c10, tx), Cuda::colorlerp(c01, c11, tx), ty);
}

__global__ void interpolate_glyph_kernel(
    const Cuda::Glyph glyph1, const Cuda::Glyph glyph2,
    const float t,
    const Cuda::vec2 tl_final, const Cuda::vec2 wh_final,
    uint32_t* d_output_pix, const Cuda::ivec2 output_wh)
{
    Cuda::vec2 grid_point(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (grid_point.x >= wh_final.x || grid_point.y >= wh_final.y) return;

    Cuda::vec2 norm = grid_point / wh_final;

    uint32_t c1 = sample_bilinear(glyph1, norm);
    uint32_t c2 = sample_bilinear(glyph2, norm);

    Cuda::ivec2 final_pos(grid_point.x + tl_final.x, grid_point.y + tl_final.y);
    overlay_pixel(final_pos, Cuda::colorlerp(c1, c2, t), 1.0, d_output_pix, output_wh);
}

__global__ void fade_glyph_kernel(
    const Cuda::Glyph glyph, const float opacity,
    const Cuda::vec2 tl_final,
    uint32_t* d_output_pix, const Cuda::ivec2 output_wh)
{
    Cuda::ivec2 grid_point(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (grid_point.x >= glyph.wh.x || grid_point.y >= glyph.wh.y) return;

    uint32_t c = glyph.pix[grid_point.y * glyph.wh.x + grid_point.x];

    Cuda::ivec2 final_pos(grid_point.x + tl_final.x, grid_point.y + tl_final.y);
    overlay_pixel(final_pos, c, opacity, d_output_pix, output_wh);
}

void interpolate_glyph(const Cuda::Interpolation& interpolation, const int g1, const int g2, const float t, uint32_t* d_output_pix, const Cuda::ivec2& output_wh)
{
    const Cuda::Glyph& glyph1 = interpolation.glyphs_1[g1];
    const Cuda::Glyph& glyph2 = interpolation.glyphs_2[g2];

    Cuda::vec2 envelope_wh = Cuda::veclerp(interpolation.wh_1, interpolation.wh_2, t);
    Cuda::vec2 tl_in_envelope = Cuda::veclerp(glyph1.top_left, glyph2.top_left, t);
    Cuda::vec2 tl_final = tl_in_envelope + (output_wh - envelope_wh) / 2.0f;
    Cuda::vec2 wh_final = Cuda::veclerp(glyph1.wh, glyph2.wh, t);

    dim3 threads(16, 16);
    dim3 blocks((wh_final.x + threads.x - 1) / threads.x, (wh_final.y + threads.y - 1) / threads.y);
    interpolate_glyph_kernel<<<blocks, threads>>>(glyph1, glyph2, t, tl_final, wh_final, d_output_pix, output_wh);
}

void fade_unmatched_glyph(const Cuda::ivec2& envelope_wh,
        const Cuda::Glyph& glyph, const float t, uint32_t* d_output_pix, const Cuda::ivec2& output_wh)
{
    dim3 threads(16, 16);
    dim3 blocks((glyph.wh.x + threads.x - 1) / threads.x, (glyph.wh.y + threads.y - 1) / threads.y);
    Cuda::vec2 tl_final = glyph.top_left + (output_wh - envelope_wh) / 2.0f;
    fade_glyph_kernel<<<blocks, threads>>>(glyph, t, tl_final, d_output_pix, output_wh);
}

// Draw every glyph at time t: matched glyphs move and morph, unmatched ones fade.
extern "C" void interpolate(
    const Cuda::Interpolation& interpolation, const float t,
    uint32_t* d_output_pix, const Cuda::ivec2 output_wh)
{
    // Matched glyphs move and morph
    for (int i = 0; i < interpolation.num_glyphs_1; i++) {
        const int j = interpolation.match_1[i];
        if (j >= 0) interpolate_glyph(interpolation, i, j, t, d_output_pix, output_wh);
    }
    // Unmatched glyphs from image 2 fade in
    for (int j = 0; j < interpolation.num_glyphs_2; j++) {
        if (interpolation.match_2[j] < 0)
            fade_unmatched_glyph(interpolation.wh_2, interpolation.glyphs_2[j], t, d_output_pix, output_wh);
    }
    // Unmatched glyphs from image 1 fade out
    for (int i = 0; i < interpolation.num_glyphs_1; i++) {
        if (interpolation.match_1[i] < 0)
            fade_unmatched_glyph(interpolation.wh_1, interpolation.glyphs_1[i], 1.0f - t, d_output_pix, output_wh);
    }
}
