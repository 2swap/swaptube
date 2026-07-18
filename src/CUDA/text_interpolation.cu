#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
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
// 3a.  Immediately discard pairs whose bounding boxes have aspect ratios that differ by more than a few percent.
// 3b.  For the remaining pairs, compute IOU (Intersection over Union) of the glyphs, and discard pairs with IOU far from 1.
//      We do a best-effort scale to make the bounding boxes of the components match before computing IOU,
//      to account for differently sized glyphs, such as exponents turning into coefficients in a derivative.
// 4.   We are now left with a bipartite graph of components, where edges represent matching glyphs.
//      We wish to eliminate as many edges as possible without leaving any glyph unpaired.
//      This is done geometrically by analyzing the positions of the components and their neighbors,
//      and eliminating edges that are inconsistent with the majority of the graph structure.
//      Mark each edge as "undecided". We will mark edges as "keep" or "discard" like so:
// 4a.  For each undecided edge e, if one of its neighbors has degree of exactly 1, mark e as "keep".
// 4b.  For each undecided edge e, if both of its neighbors have an existing "keep" edge, mark e as "discard".
//      Repeat these steps until no more edges can be marked as "keep" or "discard".
// 4c.  Score all remaining undecided edges e with position delta d,
//      by summing over all edges e' the term 1 / (10 + mag(d - d')).
//      Add the edge with the highest score to the "keep" set, and repeat from 4a.
// 5.   We now have paired shared glyphs. We can interpolate between them by linearly interpolating
//      the positions of the glyphs, and using a simple alpha blend for the pixels in the glyphs.

const float ASPECT_RATIO_THRESHOLD = 0.2f;
const float IOU_THRESHOLD = 0.85f;
const bool ALLOW_MULTI_EDGES = false;

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
    float y1 = g1.wh.y + .01f;
    float y2 = g2.wh.y + .01f;
    float aspect_ratio_1 = (float) g1.wh.x / y1;
    float aspect_ratio_2 = (float) g2.wh.x / y2;
    if (std::abs(aspect_ratio_1 - aspect_ratio_2) > ASPECT_RATIO_THRESHOLD) {
        return false;
    }
    float iou = compute_iou(g1.pix, g1.wh, g2.pix, g2.wh);
    if (iou < IOU_THRESHOLD) {
        return false;
    }
    return true;
}

Cuda::vec2 get_position_delta(const Cuda::Glyph& g1, const Cuda::Glyph& g2)
{
    // Compute the position delta between the two glyphs, which is the difference in their centers.
    return (g2.top_left + g2.wh / 2) - (g1.top_left + g1.wh / 2);
}

void construct_plausibility_matrix(Cuda::Interpolation& interpolation)
{
    Cuda::GraphAdjacencyMatrix& matrix = interpolation.adjacency_matrix;
    matrix.num_components_1 = interpolation.num_glyphs_1;
    matrix.num_components_2 = interpolation.num_glyphs_2;
    matrix.adj_matrix = new Cuda::GraphAdjacency[matrix.num_components_1 * matrix.num_components_2];
    uint32_t undecided = 0;
    uint32_t no_edge = 0;
    for (uint32_t i = 0; i < matrix.num_components_1; i++) {
        for (uint32_t j = 0; j < matrix.num_components_2; j++) {
            const uint32_t index = i * matrix.num_components_2 + j;
            const Cuda::Glyph& g1 = interpolation.glyphs_1[i];
            const Cuda::Glyph& g2 = interpolation.glyphs_2[j];
            if (is_same_glyph(g1, g2)) {
                matrix.adj_matrix[index].status = Cuda::EdgeStatus::UNDECIDED_EDGE;
                undecided++;
            } else {
                matrix.adj_matrix[index].status = Cuda::NO_EDGE;
                no_edge++;
            }
            matrix.adj_matrix[index].position_delta = get_position_delta(g1, g2);
        }
    }
}

void print_adjacency_matrix(const Cuda::GraphAdjacencyMatrix& matrix)
{
    for (uint32_t i = 0; i < matrix.num_components_1; i++) {
        for (uint32_t j = 0; j < matrix.num_components_2; j++) {
            const Cuda::GraphAdjacency& adj = matrix.adj_matrix[i * matrix.num_components_2 + j];
            const Cuda::EdgeStatus status = adj.status;
            char c = '?';
            if (status == Cuda::EdgeStatus::UNDECIDED_EDGE) c = 'U';
            else if (status == Cuda::EdgeStatus::KEEP_EDGE) c = 'K';
            else if (status == Cuda::EdgeStatus::NO_EDGE) c = '.';
            printf("%c ", c);
        }
        printf("\n");
    }
}

// Step 4a
bool add_definite_edges(Cuda::Interpolation& interpolation)
{
    bool added_edge = false;
    uint32_t added_count = 0;
    Cuda::GraphAdjacencyMatrix& matrix = interpolation.adjacency_matrix;
    const Cuda::Glyph* glyphs_1 = interpolation.glyphs_1;
    const Cuda::Glyph* glyphs_2 = interpolation.glyphs_2;
    for (uint32_t i = 0; i < matrix.num_components_1; i++) {
        for (uint32_t j = 0; j < matrix.num_components_2; j++) {
            if (matrix.adj_matrix[i * matrix.num_components_2 + j].status == Cuda::EdgeStatus::UNDECIDED_EDGE) {
                // Check if either neighbor has no edges other than this one.
                bool neighbor_1_has_no_other_options = true;
                bool neighbor_2_has_no_other_options = true;
                for (uint32_t k = 0; k < matrix.num_components_2; k++) {
                    if (k != j && matrix.adj_matrix[i * matrix.num_components_2 + k].status != Cuda::EdgeStatus::NO_EDGE) {
                        neighbor_1_has_no_other_options = false;
                        break;
                    }
                }
                for (uint32_t k = 0; k < matrix.num_components_1; k++) {
                    if (k != i && matrix.adj_matrix[k * matrix.num_components_2 + j].status != Cuda::EdgeStatus::NO_EDGE) {
                        neighbor_2_has_no_other_options = false;
                        break;
                    }
                }
                bool condition = ALLOW_MULTI_EDGES ? (neighbor_1_has_no_other_options || neighbor_2_has_no_other_options) :
                                                     (neighbor_1_has_no_other_options && neighbor_2_has_no_other_options);
                if (condition) {
                    matrix.adj_matrix[i * matrix.num_components_2 + j].status = Cuda::KEEP_EDGE;
                    matrix.adj_matrix[i * matrix.num_components_2 + j].position_delta = get_position_delta(glyphs_1[i], glyphs_2[j]);
                    added_edge = true;
                    added_count++;
                }
            }
        }
    }
    return added_edge;
}

// Step 4b
bool remove_extraneous_edges(Cuda::Interpolation& interpolation)
{
    bool removed_edge = false;
    Cuda::GraphAdjacencyMatrix& matrix = interpolation.adjacency_matrix;
    for (uint32_t i = 0; i < matrix.num_components_1; i++) {
        for (uint32_t j = 0; j < matrix.num_components_2; j++) {
            if (matrix.adj_matrix[i * matrix.num_components_2 + j].status == Cuda::EdgeStatus::UNDECIDED_EDGE) {
                // Check if both neighbors have an existing "keep" edge.
                bool neighbor_1_has_keep_edge = false;
                bool neighbor_2_has_keep_edge = false;
                for (uint32_t k = 0; k < matrix.num_components_2; k++) {
                    if (k != j && matrix.adj_matrix[i * matrix.num_components_2 + k].status == Cuda::EdgeStatus::KEEP_EDGE) {
                        neighbor_1_has_keep_edge = true;
                        break;
                    }
                }
                for (uint32_t k = 0; k < matrix.num_components_1; k++) {
                    if (k != i && matrix.adj_matrix[k * matrix.num_components_2 + j].status == Cuda::EdgeStatus::KEEP_EDGE) {
                        neighbor_2_has_keep_edge = true;
                        break;
                    }
                }
                bool condition = ALLOW_MULTI_EDGES ? (neighbor_1_has_keep_edge && neighbor_2_has_keep_edge) :
                                                     (neighbor_1_has_keep_edge || neighbor_2_has_keep_edge);
                if (condition) {
                    matrix.adj_matrix[i * matrix.num_components_2 + j].status = Cuda::EdgeStatus::NO_EDGE;
                    removed_edge = true;
                }
            }
        }
    }
    return removed_edge;
}

float score_edge(Cuda::Interpolation& interpolation, uint32_t i, uint32_t j)
{
    const Cuda::Glyph& g1 = interpolation.glyphs_1[i];
    const Cuda::Glyph& g2 = interpolation.glyphs_2[j];
    const Cuda::vec2 d = get_position_delta(g1, g2);
    float score = 0.0f;
    const Cuda::GraphAdjacencyMatrix& matrix = interpolation.adjacency_matrix;
    for (uint32_t k = 0; k < matrix.num_components_1; k++) {
        for (uint32_t l = 0; l < matrix.num_components_2; l++) {
            Cuda::GraphAdjacency& adj = matrix.adj_matrix[k * matrix.num_components_2 + l];
            if (adj.status == Cuda::EdgeStatus::KEEP_EDGE) {
                const Cuda::vec2 d_prime = adj.position_delta;
                score += 1.0f / (.1f + Cuda::length(d - d_prime));
            }
        }
    }
    return score;
}

// Step 4c
bool add_highest_score_edge(Cuda::Interpolation& interpolation)
{
    bool added_edge = false;
    float best_score = -1.0f;
    uint32_t best_i = 0;
    uint32_t best_j = 0;
    uint32_t undecided_count = 0;
    Cuda::GraphAdjacencyMatrix& matrix = interpolation.adjacency_matrix;
    const Cuda::Glyph* glyphs_1 = interpolation.glyphs_1;
    const Cuda::Glyph* glyphs_2 = interpolation.glyphs_2;
    for (uint32_t i = 0; i < matrix.num_components_1; i++) {
        for (uint32_t j = 0; j < matrix.num_components_2; j++) {
            if (matrix.adj_matrix[i * matrix.num_components_2 + j].status == Cuda::EdgeStatus::UNDECIDED_EDGE) {
                undecided_count++;
                float score = score_edge(interpolation, i, j);
                printf("Score for edge (%u, %u): %f\n", i, j, score);
                if (score > best_score) {
                    best_score = score;
                    best_i = i;
                    best_j = j;
                }
            }
        }
    }
    if (best_score > -1.0f) {
        const int index = best_i * matrix.num_components_2 + best_j;
        matrix.adj_matrix[index].status = Cuda::EdgeStatus::KEEP_EDGE;
        matrix.adj_matrix[index].position_delta = get_position_delta(glyphs_1[best_i], glyphs_2[best_j]);
        added_edge = true;
    }
    // If best_score is still -1.0f, then there are no undecided edges left, and we are done.
    return added_edge;
}

void resolve_plausibility_matrix(Cuda::Interpolation& interpolation)
{
    while (true) {
        while (true) {
            bool added = add_definite_edges(interpolation);
            bool removed = remove_extraneous_edges(interpolation);
            if (!added && !removed) break;
        }
        print_adjacency_matrix(interpolation.adjacency_matrix);
        bool added = add_highest_score_edge(interpolation);
        print_adjacency_matrix(interpolation.adjacency_matrix);
        if (!added) break;
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
    interpolation.adjacency_matrix.num_components_1 = num_glyphs_1;
    interpolation.adjacency_matrix.num_components_2 = num_glyphs_2;
    interpolation.adjacency_matrix.adj_matrix = new Cuda::GraphAdjacency[num_glyphs_1 * num_glyphs_2];

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

    construct_plausibility_matrix(interpolation);
    resolve_plausibility_matrix(interpolation);
    print_adjacency_matrix(interpolation.adjacency_matrix);

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
    delete[] interpolation.adjacency_matrix.adj_matrix;
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

// For each "keep" edge in the adjacency matrix, interpolate that glyph
extern "C" void interpolate(
    const Cuda::Interpolation& interpolation, const float t,
    uint32_t* d_output_pix, const Cuda::ivec2 output_wh)
{
    // Draw all of the glyph interpolations
    for (uint32_t i = 0; i < interpolation.adjacency_matrix.num_components_1; i++) {
        for (uint32_t j = 0; j < interpolation.adjacency_matrix.num_components_2; j++) {
            int index = i * interpolation.adjacency_matrix.num_components_2 + j;
            if (interpolation.adjacency_matrix.adj_matrix[index].status == Cuda::EdgeStatus::KEEP_EDGE) {
                interpolate_glyph(interpolation, i, j, t, d_output_pix, output_wh);
            }
        }
    }

    // Fade in any unmatched components from image 2
    for (uint32_t j = 0; j < interpolation.adjacency_matrix.num_components_2; j++) {
        bool has_edge = false;
        for (uint32_t i = 0; i < interpolation.adjacency_matrix.num_components_1; i++) {
            int index = i * interpolation.adjacency_matrix.num_components_2 + j;
            if (interpolation.adjacency_matrix.adj_matrix[index].status == Cuda::EdgeStatus::KEEP_EDGE) {
                has_edge = true;
                break;
            }
        }
        if (!has_edge) {
            fade_unmatched_glyph(interpolation.wh_2, interpolation.glyphs_2[j], t, d_output_pix, output_wh);
        }
    }

    // Fade out any unmatched components from image 1
    for (uint32_t i = 0; i < interpolation.adjacency_matrix.num_components_1; i++) {
        bool has_edge = false;
        for (uint32_t j = 0; j < interpolation.adjacency_matrix.num_components_2; j++) {
            int index = i * interpolation.adjacency_matrix.num_components_2 + j;
            if (interpolation.adjacency_matrix.adj_matrix[index].status == Cuda::EdgeStatus::KEEP_EDGE) {
                has_edge = true;
                break;
            }
        }
        if (!has_edge) {
            fade_unmatched_glyph(interpolation.wh_1, interpolation.glyphs_1[i], 1.0f - t, d_output_pix, output_wh);
        }
    }
}
