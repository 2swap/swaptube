#include <stack>
#include <cassert>
#include <unordered_set>
#include "../io/VisualMedia.cpp"

extern "C" void convolve_map_cuda(const unsigned int* a, const int aw, const int ah, const unsigned int* b, const int bw, const int bh, unsigned int* map, const int mapw, const int maph);

class TranslatedPixels {
public:
    int translation_x;
    int translation_y;
    Pixels pixels;

    TranslatedPixels(int width, int height, int tx, int ty)
        : translation_x(tx), translation_y(ty), pixels(width, height) {}

    TranslatedPixels(const Pixels& p, int tx, int ty)
        : translation_x(tx), translation_y(ty), pixels(p) {}

    TranslatedPixels(const TranslatedPixels& p, int tx, int ty)
        : translation_x(tx+p.translation_x), translation_y(ty+p.translation_y), pixels(p.pixels) {}

    inline bool out_of_range(int x, int y) const {
        return pixels.out_of_range(x - translation_x, y - translation_y);
    }

    inline int get_pixel(int x, int y) const {
        if (out_of_range(x, y)) return 0;
        return pixels.get_pixel(x - translation_x, y - translation_y);
    }

    inline void set_pixel(int x, int y, int col) {
        if (out_of_range(x, y)) return;
        pixels.set_pixel(x - translation_x, y - translation_y, col);
    }

    inline int get_alpha(int x, int y) const {
        return geta(get_pixel(x, y));
    }

    inline bool is_empty() const {
        return pixels.is_empty();
    }
};

Pixels convolve_map(const Pixels& p1, const Pixels& p2, int& max_x, int& max_y) {
    int mapw = p1.w + p2.w - 1;
    int maph = p1.h + p2.h - 1;
    vector<unsigned int> map(maph * mapw);

    // Perform the convolution using CUDA
    convolve_map_cuda(p1.pixels.data(), p1.w, p1.h, p2.pixels.data(), p2.w, p2.h, map.data(), mapw, maph);

    vector<vector<unsigned int>> map_2d(maph, vector<unsigned int>(mapw, -1));
    
    // Initialize max_value to the minimum possible value
    unsigned int max_value = 0u;

    // Copy data from the flat map to the 2D map_2d and find the max value coordinates
    for (int y = 0; y < maph; ++y) {
        for (int x = 0; x < mapw; ++x) {
            unsigned int value = map[y * mapw + x];
            map_2d[y][x] = value;

            // Check if this value is the new maximum
            if (value > max_value) {
                max_value = value;
                max_x = x;
                max_y = y;
            }
        }
    }

    max_x -= p2.w - 1;
    max_y -= p2.h - 1;

    // Create the Pixels object from the intensity map
    Pixels p = create_alpha_from_intensities(map_2d);
    p.print_to_terminal();
    
    return p;
}

void flood_fill(TranslatedPixels& ret, const TranslatedPixels& p, int start_x, int start_y, int color) {
    stack<pair<int, int>> stack;
    stack.push({start_x, start_y});

    while (!stack.empty()) {
        auto [x, y] = stack.top();
        stack.pop();

        if (p.out_of_range(x, y) || p.get_alpha(x, y) == 0 || ret.get_pixel(x, y) != 0)
            continue;

        ret.set_pixel(x, y, color);

        stack.push({x + 1, y}); // Right
        stack.push({x - 1, y}); // Left
        stack.push({x, y + 1}); // Down
        stack.push({x, y - 1}); // Up
    }
}

TranslatedPixels segment(const TranslatedPixels& p, int& id) {
    TranslatedPixels ret(Pixels(p.pixels.w, p.pixels.h), p.translation_x, p.translation_y);

    id = 0;

    // Perform flood fill for each pixel in the input TranslatedPixels
    for (int y = 0; y < p.pixels.h; y++) {
        for (int x = 0; x < p.pixels.w; x++) {
            if (p.get_alpha(x, y) != 0 && ret.get_pixel(x, y) == 0) {
                id++; // Increment the identifier for the next shape
                flood_fill(ret, p, x, y, id);
            }
        }
    }

    return ret;
}

void flood_fill_connected_to_opaque(const Pixels& p, Pixels& connected_to_opaque, int x, int y) {
    int width = p.w;
    int height = p.h;

    // Check if current pixel is within bounds and has nonzero alpha channel
    if (x < 0 || x >= width || y < 0 || y >= height || p.get_alpha(x, y) == 0)
        return;

    // Check if the current pixel is already marked in connected_to_opaque
    if (connected_to_opaque.get_pixel(x, y) != 0)
        return;

    // Mark the current pixel as connected to an opaque pixel
    connected_to_opaque.set_pixel(x, y, 0xffffffff);

    // Recursive flood fill for adjacent pixels
    flood_fill_connected_to_opaque(p, connected_to_opaque, x + 1, y); // Right
    flood_fill_connected_to_opaque(p, connected_to_opaque, x - 1, y); // Left
    flood_fill_connected_to_opaque(p, connected_to_opaque, x, y + 1); // Down
    flood_fill_connected_to_opaque(p, connected_to_opaque, x, y - 1); // Up
}

Pixels remove_unconnected_components(const Pixels& p) {
    int width = p.w;
    int height = p.h;

    // Create a Pixels object to track if each pixel is connected to an opaque pixel
    Pixels output(width, height);
    output.fill(0);

    // Iterate over each pixel in the input image
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Check if current pixel has alpha channel high enough and is not marked as connected_to_opaque
            if (p.get_alpha(x, y) > 50 && output.get_pixel(x, y) == 0) {
                // Perform flood-fill to mark all connected pixels as connected to an opaque pixel
                flood_fill_connected_to_opaque(p, output, x, y);
            }
        }
    }

    // Construct the output by copying the input and bitwise AND-ing it
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixel = p.get_pixel(x, y) & output.get_pixel(x, y);
            output.set_pixel(x, y, pixel);
        }
    }

    return output;
}

TranslatedPixels intersect(const TranslatedPixels& tp1, const TranslatedPixels& tp2) {
    // Determine the intersection boundaries
    int x_min = max(tp1.translation_x, tp2.translation_x);
    int y_min = max(tp1.translation_y, tp2.translation_y);
    int x_max = min(tp1.translation_x + tp1.pixels.w, tp2.translation_x + tp2.pixels.w);
    int y_max = min(tp1.translation_y + tp1.pixels.h, tp2.translation_y + tp2.pixels.h);

    // Calculate the width and height of the resulting intersection
    int intersection_width = x_max - x_min;
    int intersection_height = y_max - y_min;

    // If there's no intersection, return an empty TranslatedPixels
    if (intersection_width <= 0 || intersection_height <= 0) {
        return TranslatedPixels(Pixels(0, 0), 0, 0);
    }

    // Create an empty TranslatedPixels object to store the intersection result
    TranslatedPixels result(Pixels(intersection_width, intersection_height), x_min, y_min);

    // Iterate over the intersection area and set the pixel values in the result
    for (int y = y_min; y < y_max; ++y) {
        for (int x = x_min; x < x_max; ++x) {
            // Get the pixels from both TranslatedPixels objects
            int pixel1 = tp1.get_pixel(x, y);
            int pixel2 = tp2.get_pixel(x, y);

            // Select the pixel with the lower alpha value
            if (geta(pixel1) > geta(pixel2)) {
                result.set_pixel(x, y, pixel2);
            } else {
                result.set_pixel(x, y, pixel1);
            }
        }
    }

    return result;
}

TranslatedPixels unify(const TranslatedPixels& tp1, const TranslatedPixels& tp2) {
    // Determine the union boundaries
    int x_min = min(tp1.translation_x, tp2.translation_x);
    int y_min = min(tp1.translation_y, tp2.translation_y);
    int x_max = max(tp1.translation_x + tp1.pixels.w, tp2.translation_x + tp2.pixels.w);
    int y_max = max(tp1.translation_y + tp1.pixels.h, tp2.translation_y + tp2.pixels.h);

    // Calculate the width and height of the resulting union
    int union_width = x_max - x_min;
    int union_height = y_max - y_min;

    // Create an empty TranslatedPixels object to store the union result
    TranslatedPixels result(Pixels(union_width, union_height), x_min, y_min);

    // Iterate over the union area and set the pixel values in the result
    for (int y = y_min; y < y_max; ++y) {
        for (int x = x_min; x < x_max; ++x) {
            // Get the pixels from both TranslatedPixels objects
            int pixel1 = tp1.get_pixel(x, y);
            int pixel2 = tp2.get_pixel(x, y);

            // Select the pixel with the higher alpha value
            if (geta(pixel1) > geta(pixel2)) {
                result.set_pixel(x, y, pixel1);
            } else {
                result.set_pixel(x, y, pixel2);
            }
        }
    }

    return result;
}

TranslatedPixels subtract(const TranslatedPixels& original, const TranslatedPixels& to_subtract) {
    // Create a new TranslatedPixels object to store the subtraction result
    TranslatedPixels result(Pixels(original.pixels.w, original.pixels.h), original.translation_x, original.translation_y);

    // Iterate over the entire frame of the original TranslatedPixels
    int x_start = original.translation_x;
    int y_start = original.translation_y;
    int x_end = original.translation_x + original.pixels.w;
    int y_end = original.translation_y + original.pixels.h;

    for (int y = y_start; y < y_end; y++) {
        for (int x = x_start; x < x_end; x++) {
            int pixel1 = original.get_pixel(x, y);
            int pixel2 = to_subtract.get_pixel(x, y);  // This will handle out-of-bounds internally

            int alpha_diff = geta(pixel1) - geta(pixel2);
            int new_alpha = max(alpha_diff, 0);
            int new_pixel = argb_to_col(new_alpha, getr(pixel1), getg(pixel1), getb(pixel1));

            result.set_pixel(x, y, new_pixel);
        }
    }

    return result;
}

struct StepResult {
    int max_x;
    int max_y;
    Pixels map;
    Pixels induced1;
    Pixels induced2;
    Pixels current_p1;
    Pixels current_p2;
    Pixels intersection;

    StepResult(int mx, int my, Pixels cm, Pixels i1, Pixels i2, Pixels p1, Pixels p2, Pixels i)
            : max_x(mx), max_y(my), map(cm), induced1(i1), induced2(i2), current_p1(p1), current_p2(p2), intersection(i) {}
};

void flood_fill_copy_shape(const TranslatedPixels& source, TranslatedPixels& destination, int start_x, int start_y) {
    stack<pair<int, int>> stack;
    stack.push({start_x, start_y});

    while (!stack.empty()) {
        auto [sx, sy] = stack.top();
        stack.pop();

        if (source.out_of_range(sx, sy) || destination.out_of_range(sx, sy))
            continue;

        if (source.get_alpha(sx, sy) == 0 || destination.get_pixel(sx, sy) == source.get_pixel(sx, sy))
            continue;

        destination.set_pixel(sx, sy, source.get_pixel(sx, sy));

        stack.push({sx + 1, sy}); // Right
        stack.push({sx - 1, sy}); // Left
        stack.push({sx, sy + 1}); // Down
        stack.push({sx, sy - 1}); // Up
    }
}

TranslatedPixels induce(const TranslatedPixels& original, const TranslatedPixels& intersection) {
    // Create a new TranslatedPixels object with the same frame as the original
    TranslatedPixels induced(Pixels(original.pixels.w, original.pixels.h), original.translation_x, original.translation_y);

    // Determine the bounds of the intersection
    int x_start = intersection.translation_x;
    int y_start = intersection.translation_y;
    int x_end = intersection.translation_x + intersection.pixels.w;
    int y_end = intersection.translation_y + intersection.pixels.h;

    // Copy the intersection pixels to the induced TranslatedPixels
    for (int y = y_start; y < y_end; y++) {
        for (int x = x_start; x < x_end; x++) {
            int intersection_pixel = intersection.get_pixel(x, y);
            if (intersection.get_alpha(x, y) != 0) {
                induced.set_pixel(x, y, intersection_pixel);
            }
        }
    }

    // Perform flood fill to copy the corresponding shapes from the original to the induced TranslatedPixels
    for (int y = y_start; y < y_end; y++) {
        for (int x = x_start; x < x_end; x++) {
            if (intersection.get_alpha(x, y) != 0 && induced.get_pixel(x, y) != original.get_pixel(x, y)) {
                // Found a new shape, perform flood fill starting from the current pixel in the original
                flood_fill_copy_shape(original, induced, x, y);
            }
        }
    }

    return induced;
}

TranslatedPixels erase_low_iou(const TranslatedPixels& intersection, const TranslatedPixels& unified, float threshold) {
    int intersection_id = 0;
    int union_id = 0;

    // Segment both the intersection and the union to identify connected components
    TranslatedPixels segmented_intersection = segment(intersection, intersection_id);
    TranslatedPixels segmented_union = segment(unified, union_id);

    // Create a 2D vector to hold the overlap areas between intersection and union segments
    vector<vector<int>> overlap_areas(union_id + 1, vector<int>(intersection_id + 1, 0));
    vector<int> union_areas(union_id + 1, 0);
    vector<int> intersection_areas(intersection_id + 1, 0);

    // Calculate the bounds for unified
    int x_start = 0;  // Local coordinates in unified's frame
    int y_start = 0;  // Local coordinates in unified's frame
    int x_end = unified.pixels.w;
    int y_end = unified.pixels.h;

    // Iterate over the bounds of the unified pixels
    for (int y = y_start; y < y_end; ++y) {
        for (int x = x_start; x < x_end; ++x) {
            int union_segment_id = segmented_union.get_pixel(x, y);
            int intersection_segment_id = segmented_intersection.get_pixel(x, y);

            if (union_segment_id > 0) {
                union_areas[union_segment_id]++;
                if (intersection_segment_id > 0) {
                    overlap_areas[union_segment_id][intersection_segment_id]++;
                    intersection_areas[intersection_segment_id]++;
                }
            }
        }
    }

    // Identify the set of components to keep based on IoU threshold
    unordered_set<int> components_to_keep;
    for (int intersection_segment = 1; intersection_segment <= intersection_id; ++intersection_segment) {
        for (int union_segment = 1; union_segment <= union_id; ++union_segment) {
            float iou = static_cast<float>(overlap_areas[union_segment][intersection_segment]) / 
                        (union_areas[union_segment] + intersection_areas[intersection_segment] - overlap_areas[union_segment][intersection_segment]);
            if (iou >= threshold) {
                components_to_keep.insert(intersection_segment);
                break;
            }
        }
    }

    // Create a result TranslatedPixels object initialized with zero transparency
    TranslatedPixels result(Pixels(intersection.pixels.w, intersection.pixels.h), intersection.translation_x, intersection.translation_y);

    // Copy the pixels from the intersection to the result if they belong to a component to keep
    for (int y = 0; y < segmented_intersection.pixels.h; ++y) {
        for (int x = 0; x < segmented_intersection.pixels.w; ++x) {
            int intersection_segment_id = segmented_intersection.get_pixel(x, y);
            if (components_to_keep.count(intersection_segment_id) > 0) {
                result.set_pixel(x, y, intersection.get_pixel(x, y));
            }
        }
    }

    return result;
}

vector<StepResult> find_intersections(const Pixels& p1, const Pixels& p2) {
    string convolution_name = to_string(rand());
    cout << "Attempting convolution " << convolution_name << endl;
    vector<StepResult> results;

    Pixels current_p1(p1);
    Pixels current_p2(p2);
    int i = 0;

    bool intersection_nonempty = 1;
    while (intersection_nonempty) {
        i++;
        int max_x = 0, max_y = 0;

        // Perform convolution mapping to find maximum intersection
        const Pixels cm = convolve_map(current_p1, current_p2, max_x, max_y);

cout << max_x << ", " << max_y << endl;
        const TranslatedPixels translated_p1(current_p1,     0,     0);
        const TranslatedPixels translated_p2(current_p2, max_x, max_y);

        // Intersect the two TranslatedPixels objects based on the maximum convolution
        const TranslatedPixels intersection = intersect(translated_p1, translated_p2);
        const TranslatedPixels unified      =     unify(translated_p1, translated_p2);
        const TranslatedPixels erasure      = erase_low_iou(intersection, unified, .4);

        intersection_nonempty = !erasure.is_empty();

        const TranslatedPixels induced1 = induce(translated_p1, erasure);
        const TranslatedPixels induced2 = induce(translated_p2, erasure);

        // Subtract the intersection from the starting Pixels
        current_p1 = remove_unconnected_components(subtract(translated_p1, induced1).pixels);
        current_p2 = remove_unconnected_components(subtract(translated_p2, induced2).pixels);

        // Store the result of this step
        StepResult step_result(max_x, max_y, cm, induced1.pixels, induced2.pixels, current_p1, current_p2, erasure.pixels);
        results.push_back(step_result);

        // DEBUG
        ensure_dir_exists(PATH_MANAGER.this_run_output_dir + convolution_name);
        pix_to_png(cm                 , convolution_name + "/convolution_" + to_string(i) + "_a_map");
        pix_to_png(intersection.pixels, convolution_name + "/convolution_" + to_string(i) + "_b_intersection");
        pix_to_png(unified.pixels     , convolution_name + "/convolution_" + to_string(i) + "_c_unified");
        pix_to_png(erasure.pixels     , convolution_name + "/convolution_" + to_string(i) + "_d_erasure");
        pix_to_png(induced1.pixels    , convolution_name + "/convolution_" + to_string(i) + "_e_induced1");
        pix_to_png(induced2.pixels    , convolution_name + "/convolution_" + to_string(i) + "_f_induced2");
        pix_to_png(current_p1         , convolution_name + "/convolution_" + to_string(i) + "_g_p1");
        pix_to_png(current_p2         , convolution_name + "/convolution_" + to_string(i) + "_h_p2");
    }

    return results;
}
