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
// 1.   (CPU, before staging) Segment each image into glyphs: every pixel is labeled with the uint32_t index
//      (starting at 1) of the glyph it belongs to, 0 for background. Glyphs come from the LaTeX SVG's own
//      per-symbol structure (see segment_by_glyph_ids in Core/GlyphSegmentation.cpp), so multi-part symbols like
//      i, j and = are single glyphs.
// 2.   (CPU, before staging) Identify each glyph: a shape class read from the LaTeX SVG (see latex_glyph_shapes in
//      IO/Latex.cpp), equal for the same symbol at any size and resolution. Two glyphs can pair iff their classes match.
// 3.   We are now left with a bipartite graph of glyphs, where edges connect glyphs of the same class.
//      Choose the one-to-one matching with the least total on-screen travel distance, solved using the
//      Hungarian algorithm. Unmatched leftover glyphs fade out or in.
// 4.   We now have paired shared glyphs. We can interpolate between them by linearly interpolating
//      the positions of the glyphs, and using a simple alpha blend for the pixels in the glyphs.

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

__global__ void copy_glyph_to_gpu_kernel(const uint32_t* pix, const uint32_t* vis, const Cuda::ivec2 wh, const Cuda::ivec2 top_left, const Cuda::ivec2 bottom_right, uint32_t* d_glyph_pix, const uint32_t label, const uint32_t transparent_color)
{
    Cuda::ivec2 grid_point = Cuda::ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const Cuda::ivec2 glyph_wh = bottom_right - top_left;
    if (grid_point.x >= glyph_wh.x || grid_point.y >= glyph_wh.y) return;
    Cuda::ivec2 pix_point = top_left + grid_point;

    const int index = pix_point.y * wh.x + pix_point.x;
    d_glyph_pix[grid_point.y * glyph_wh.x + grid_point.x] = pix[index] == label ? vis[index] : transparent_color;
}

// Picks one color per glyph, for its transparent pixels: the RGB of the glyph's most opaque pixel, with zero alpha.
// Bilinear sampling lerps RGB even where alpha is 0, so transparent pixels must carry the glyph's own color,
// or its edges would blend toward some other color while it moves.
// Indexed by label, so entry 0 (background) is unused.
static std::vector<uint32_t> transparent_glyph_colors(const uint32_t* labels, const uint32_t* vis, const Cuda::ivec2 wh, const int num_glyphs)
{
    std::vector<uint32_t> colors(num_glyphs + 1, 0x00FFFFFF);
    std::vector<int> best_alpha(num_glyphs + 1, -1);
    for (int p = 0; p < wh.x * wh.y; p++) {
        const uint32_t label = labels[p];
        if (label == 0 || label > (uint32_t) num_glyphs) continue;
        const int alpha = Cuda::geta(vis[p]);
        if (alpha > best_alpha[label]) {
            best_alpha[label] = alpha;
            colors[label] = vis[p] & 0x00FFFFFF;
        }
    }
    return colors;
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

// Step 3: among the pairs whose shape classes match, choose the matching with the most
// matched pairs, breaking ties by least total travel distance. Writes the result to match_1 and match_2.
// Solved as a square n x n assignment problem with n = max(n1, n2): rows are image 1 glyphs, columns are image 2
// glyphs, and the shorter side is padded with free dummies.
// shapes_1[i] and shapes_2[j] are the glyphs' shape classes; 0 means unknown, which never matches.
void solve_matching(Cuda::Interpolation& interpolation, const uint32_t* shapes_1, const uint32_t* shapes_2)
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
            cost[(size_t) i * n + j] = (shapes_1[i] != 0 && shapes_1[i] == shapes_2[j])
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
// The CPU is expected to have done steps 1 and 2 already: h_pix are glyph label images, and h_shapes[i] is the
// shape class of glyph i (label i + 1).
extern "C" Cuda::Interpolation stage_interpolation(
    const uint32_t* h_pix_1, const uint32_t* h_vis_1, const uint32_t* h_shapes_1, const Cuda::ivec2 wh_1, const int num_glyphs_1,
    const uint32_t* h_pix_2, const uint32_t* h_vis_2, const uint32_t* h_shapes_2, const Cuda::ivec2 wh_2, const int num_glyphs_2)
{
    Cuda::Interpolation interpolation;

    interpolation.wh_1 = wh_1;
    interpolation.wh_2 = wh_2;

    cudaMalloc(&interpolation.pix_1, wh_1.x * wh_1.y * sizeof(uint32_t));
    cudaMalloc(&interpolation.pix_2, wh_2.x * wh_2.y * sizeof(uint32_t));
    cudaMemcpy(interpolation.pix_1, h_pix_1, wh_1.x * wh_1.y * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(interpolation.pix_2, h_pix_2, wh_2.x * wh_2.y * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // The rendered (colored) images
    uint32_t* d_vis_1;
    uint32_t* d_vis_2;
    cudaMalloc(&d_vis_1, wh_1.x * wh_1.y * sizeof(uint32_t));
    cudaMalloc(&d_vis_2, wh_2.x * wh_2.y * sizeof(uint32_t));
    cudaMemcpy(d_vis_1, h_vis_1, wh_1.x * wh_1.y * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vis_2, h_vis_2, wh_2.x * wh_2.y * sizeof(uint32_t), cudaMemcpyHostToDevice);

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

    const std::vector<uint32_t> transparent_colors_1 = transparent_glyph_colors(h_pix_1, h_vis_1, wh_1, num_glyphs_1);
    const std::vector<uint32_t> transparent_colors_2 = transparent_glyph_colors(h_pix_2, h_vis_2, wh_2, num_glyphs_2);

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
        copy_glyph_to_gpu_kernel<<<blocks, threads>>>(interpolation.pix_1, d_vis_1, wh_1, top_left, bottom_right, interpolation.glyphs_1[i].pix, i + 1, transparent_colors_1[i + 1]);
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
        copy_glyph_to_gpu_kernel<<<blocks, threads>>>(interpolation.pix_2, d_vis_2, wh_2, top_left, bottom_right, interpolation.glyphs_2[i].pix, i + 1, transparent_colors_2[i + 1]);
    }

    cudaDeviceSynchronize();
    cudaFree(d_top_left);
    cudaFree(d_bottom_right);
    cudaFree(d_vis_1);
    cudaFree(d_vis_2);

    solve_matching(interpolation, h_shapes_1, h_shapes_2);
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
    const float x = uv.x * glyph.wh.x - 0.5f;
    const float y = uv.y * glyph.wh.y - 0.5f;
    const float fx = floorf(x), fy = floorf(y);
    const float tx = x - fx;
    const float ty = y - fy;

    const int x0 = max(0, min(glyph.wh.x - 1, (int) fx));
    const int y0 = max(0, min(glyph.wh.y - 1, (int) fy));
    const int x1 = max(0, min(glyph.wh.x - 1, (int) fx + 1));
    const int y1 = max(0, min(glyph.wh.y - 1, (int) fy + 1));

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

    // Sample at this output pixel's center
    Cuda::vec2 norm = Cuda::vec2(grid_point.x + 0.5f, grid_point.y + 0.5f) / wh_final;

    uint32_t c1 = sample_bilinear(glyph1, norm);
    uint32_t c2 = sample_bilinear(glyph2, norm);

    // Round to the nearest pixel: exact at t = 0 and 1, where tl_final is whole, and unbiased in between
    Cuda::ivec2 final_pos((int) floorf(grid_point.x + tl_final.x + 0.5f), (int) floorf(grid_point.y + tl_final.y + 0.5f));
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

Cuda::ivec2 static_top_left(const Cuda::ivec2& output_wh, const Cuda::ivec2& image_wh)
{
    const Cuda::vec2 center(output_wh.x / 2.0f, output_wh.y / 2.0f);
    return Cuda::floor(center - (image_wh * 0.5f));
}

void interpolate_glyph(const Cuda::Interpolation& interpolation, const int g1, const int g2, const float t, uint32_t* d_output_pix, const Cuda::ivec2& output_wh)
{
    const Cuda::Glyph& glyph1 = interpolation.glyphs_1[g1];
    const Cuda::Glyph& glyph2 = interpolation.glyphs_2[g2];

    // Move between the glyph's exact on-screen positions in the two static frames
    const Cuda::vec2 screen_tl_1 = glyph1.top_left + static_top_left(output_wh, interpolation.wh_1);
    const Cuda::vec2 screen_tl_2 = glyph2.top_left + static_top_left(output_wh, interpolation.wh_2);
    Cuda::vec2 tl_final = Cuda::veclerp(screen_tl_1, screen_tl_2, t);
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
    Cuda::vec2 tl_final = glyph.top_left + static_top_left(output_wh, envelope_wh);
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
