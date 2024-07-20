#include <stack>

int convolve(const Pixels& a, const Pixels& b, int dx, int dy){
    /*a should be smaller for speed*/
    if(a.w*a.h>b.w*b.h) return convolve(b, a, -dx, -dy);

    double sum = 0;
    int minx = max(0, dx);
    int miny = max(0, dy);
    int maxx = min(a.w, b.w+dx);
    int maxy = min(a.h, b.h+dy);

    int jump = 2;
    for (int x = minx; x < maxx; x+=jump)
        for (int y = miny; y < maxy; y+=jump){
            sum += (a.get_alpha(x, y) > 128) && (b.get_alpha(x-dx, y-dy) > 128);
        }
    return sum;
}

void shrink_alpha_from_center(Pixels& p) {
    int centerX = p.w / 2;
    int centerY = p.h / 2;

    for (int y = 0; y < p.h; y++) {
        for (int x = 0; x < p.w; x++) {
            int distanceX = centerX - x;
            int distanceY = centerY - y;
            double distance = sqrt(distanceX * distanceX + distanceY * distanceY);

            int alpha = p.get_alpha(x, y);
            int shrunkAlpha = static_cast<int>(alpha * (1.0 - distance / std::max(centerX, centerY)));

            p.set_alpha(x, y, shrunkAlpha);
        }
    }
}

Pixels convolve_map(const Pixels& p1, const Pixels& p2, int& max_x, int& max_y){
    int max_conv = 0;
    int retw = p1.w+p2.w;
    int reth = p1.h+p2.h;
    vector<vector<int>> map(reth, vector<int>(retw, -1));
    int jump = 3;
    for(int x = retw/4; x < 3*retw/4; x+=jump)
        for(int y = reth/4; y < 3*reth/4; y+=jump){
            int convolution = convolve(p1, p2, x-p2.w, y-p2.h);
            if(convolution > max_conv){
                max_conv = convolution;
                max_x = x;
                max_y = y;
            }
            for(int dx = 0; dx < jump; dx++)
                for(int dy = 0; dy < jump; dy++)
                    map[y+dy][x+dx] = convolution;
        }
    for(int x = max(0, max_x-jump); x < min(retw, max_x+jump); x++)
        for(int y = max(0, max_y-jump); y < min(reth, max_y+jump); y++){
            int convolution = convolve(p1, p2, x-p2.w, y-p2.h);
            if(convolution > max_conv){
                max_conv = convolution;
                max_x = x;
                max_y = y;
            }
            map[y][x] = convolution;
        }
    max_x -= p2.w;
    max_y -= p2.h;

    return create_alpha_from_intensities(map, 0);
}

/**
 * Fills the connected region in the output Pixels object with the specified color
 * using the flood fill algorithm, starting from the given starting coordinates.
 *
 * @param ret The output Pixels object to fill.
 * @param p The input Pixels object to perform the flood fill on.
 * @param start_x The starting x-coordinate of the fill.
 * @param start_y The starting y-coordinate of the fill.
 * @param color The color value to assign to the filled region.
 */
void flood_fill(Pixels& ret, const Pixels& p, int start_x, int start_y, int color) {
    stack<pair<int, int> > stack;
    stack.push({start_x, start_y});

    while (!stack.empty()) {
        auto [x, y] = stack.top();
        stack.pop();

        if (x < 0 || x >= p.w || y < 0 || y >= p.h)
            continue;

        if (p.get_alpha(x, y) == 0 || ret.get_pixel(x, y) != 0)
            continue;

        ret.set_pixel(x, y, color);

        stack.push({x + 1, y}); // Right
        stack.push({x - 1, y}); // Left
        stack.push({x, y + 1}); // Down
        stack.push({x, y - 1}); // Up
    }
}

int connected_component_size(const Pixels& p, int start_x, int start_y) {
    int component_size = 0;
    int target_alpha = p.get_alpha(start_x, start_y);
    Pixels visited(p.w, p.h);

    stack<pair<int, int>> stack;
    stack.push({start_x, start_y});

    while (!stack.empty()) {
        auto [x, y] = stack.top();
        stack.pop();

        if (x < 0 || x >= p.w || y < 0 || y >= p.h)
            continue;

        if (p.get_alpha(x, y) != target_alpha || visited.get_pixel(x, y) != 0)
            continue;

        visited.set_pixel(x, y, 1);
        component_size++;

        stack.push({x + 1, y}); // Right
        stack.push({x - 1, y}); // Left
        stack.push({x, y + 1}); // Down
        stack.push({x, y - 1}); // Up
    }

    return component_size;
}

Pixels segment(const Pixels& p, int& id) {
    Pixels ret(p.w, p.h);

    // Initialize the identifier color
    id = 0;

    // Perform flood fill for each pixel in the input Pixels
    for (int y = 0; y < p.h; y++) {
        for (int x = 0; x < p.w; x++) {
            if (p.get_alpha(x, y) != 0 && ret.get_pixel(x, y) == 0) {
                id++; // Increment the identifier color for the next shape
                // Found a new shape, perform flood fill starting from the current pixel
                flood_fill(ret, p, x, y, id);
            }
        }
    }

    return ret;
}

Pixels erase_small_components(const Pixels& p, int min_size) {
    int id = 0;
    Pixels segmented = segment(p, id);

    // Calculate the area of each segmented component
    vector<int> componentArea(id, 0);
    for (int y = 0; y < segmented.h; y++) {
        for (int x = 0; x < segmented.w; x++) {
            int componentId = segmented.get_pixel(x, y);
            if (componentId != 0) {
                componentArea[componentId - 1]++;
            }
        }
    }

    // Erase components with area less than 'min_size' pixels
    Pixels result = p;
    for (int y = 0; y < segmented.h; y++) {
        for (int x = 0; x < segmented.w; x++) {
            int componentId = segmented.get_pixel(x, y);
            if (componentArea[componentId - 1] < min_size) {
                result.set_pixel(x, y, argb_to_col(0,255,255,255));
            }
        }
    }

    return result;
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

Pixels intersect(const Pixels& p1, const Pixels& p2, int dx, int dy) {
    Pixels result(p1.w, p1.h);

    for (int y = 0; y < p1.h; y++) {
        for (int x = 0; x < p1.w; x++) {
            int pixel1 = p1.get_pixel(x, y);
            int pixel2 = p2.get_pixel(x - dx, y - dy);

            if (geta(pixel1) > geta(pixel2)) {
                result.set_pixel(x, y, pixel2);
            } else {
                result.set_pixel(x, y, pixel1);
            }
        }
    }

    return result;
}

Pixels subtract(const Pixels& p1, const Pixels& p2, int dx, int dy) {
    Pixels result(p1.w, p1.h);

    for (int y = 0; y < p1.h; y++) {
        for (int x = 0; x < p1.w; x++) {
            int pixel1 = p1.get_pixel(x, y);
            int pixel2 = p2.get_pixel(x - dx, y - dy);

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

void flood_fill_copy_shape(const Pixels& from, Pixels& to, int from_x, int from_y, int to_x, int to_y) {
    stack<tuple<int, int, int, int>> stack;
    stack.push({from_x, from_y, to_x, to_y});

    while (!stack.empty()) {
        auto [fx, fy, tx, ty] = stack.top();
        stack.pop();

        if (fx < 0 || fx >= from.w || fy < 0 || fy >= from.h)
            continue;

        if (tx < 0 || tx >= to.w || ty < 0 || ty >= to.h)
            continue;

        if (from.get_alpha(fx, fy) == 0 || to.get_pixel(tx, ty) == from.get_pixel(fx, fy))
            continue;

        to.set_pixel(tx, ty, from.get_pixel(fx, fy));

        stack.push({fx + 1, fy, tx + 1, ty}); // Right
        stack.push({fx - 1, fy, tx - 1, ty}); // Left
        stack.push({fx, fy + 1, tx, ty + 1}); // Down
        stack.push({fx, fy - 1, tx, ty - 1}); // Up
    }
}

Pixels induce(const Pixels& from, const Pixels& to, int dx, int dy) {
    Pixels induced(2*from.w+to.w, 2*from.h+to.h);

    // Copy the to Pixels to the induced Pixels
    for (int y = 0; y < to.h; y++) {
        for (int x = 0; x < to.w; x++) {
            int pixel = to.get_pixel(x, y);
            induced.set_pixel(x+from.w, y+from.h, pixel);
        }
    }

    // Perform flood fill for each pixel in the translated to Pixels
    for (int y = 0; y < to.h; y++) {
        for (int x = 0; x < to.w; x++) {
            int from_x = x+dx;
            int from_y = y+dy;
            int to_x = x;
            int to_y = y;
            int induced_x = x+from.w;
            int induced_y = y+from.h;
            if (to.get_alpha(x, y) != 0 && induced.get_pixel(induced_x, induced_y) != from.get_pixel(from_x, from_y)) {
                // Found a new shape, perform flood fill starting from the current pixel in from
                flood_fill_copy_shape(from, induced, from_x, from_y, induced_x, induced_y);
            }
        }
    }

    return induced;
}

Pixels erase_low_iou(const Pixels& intersection, float threshold, const Pixels& p1, const Pixels& p2, int max_x, int max_y) {
    int id1 = 0;
    Pixels segmented1 = segment(p1, id1);

    int id2 = 0;
    Pixels segmented2 = segment(p2, id2);

    int id3 = 0;
    Pixels segmented3 = segment(intersection, id3);

    int iu_w = id1+id2;
    int iu_h = id3;

    vector<vector<int>> intersected_pixels(iu_h, vector<int>(iu_w, 0));
    vector<vector<int>> united_pixels(iu_h, vector<int>(iu_w, 0));

    // Find unions and intersections

    // Loop over coordinates
    for (int y = 0; y < intersection.h; y++) {
        for (int x = 0; x < intersection.w; x++) {
            int componentId1 = segmented1.get_pixel(x, y) - 1;
            int componentId2 = segmented2.get_pixel(x - max_x, y - max_y) - 1;
            int componentId3 = segmented3.get_pixel(x, y) - 1;

            if (componentId1 != -1) {
                // Loop over all segmented3 components
                for (int y = 0; y < iu_h; y++) {
                    united_pixels[y][componentId1]++;
                }
            }

            if (componentId2 != -1) {
                // Loop over all segmented3 components
                for (int y = 0; y < iu_h; y++) {
                    united_pixels[y][componentId2 + id1]++;
                }
            }

            if (componentId3 != -1) {
                // Loop over all segmented1/2 components
                for (int x = 0; x < iu_w; x++) {
                    united_pixels[componentId3][x]++;
                }
            }

            if (componentId3 != -1 && componentId1 != -1) {
                intersected_pixels[componentId3][componentId1]++;
                united_pixels[componentId3][componentId1]--;
            }

            if (componentId3 != -1 && componentId2 != -1) {
                intersected_pixels[componentId3][componentId2 + id1]++;
                united_pixels[componentId3][componentId2 + id1]--;
            }
        }
    }

    // Identify low IOU components

    // Create the result Pixels and copy the intersection
    Pixels result = intersection;

    // Vector to store the decision of keeping or erasing each component
    vector<bool> keepComponent(iu_h, false);

    // Iterate over each component in segmented3
    for (int componentId3 = 0; componentId3 < iu_h; componentId3++) {
        // Check IoU with components in p1
        for (int x = 0; x < iu_w; x++) {
            float iou = static_cast<float>(intersected_pixels[componentId3][x]) /
                        united_pixels[componentId3][x];
            if (iou >= threshold) {
                keepComponent[componentId3] = true;
                break;
            }
        }
    }

    // Erase the low-IOU components from the result
    for (int y = 0; y < segmented3.h; y++) {
        for (int x = 0; x < segmented3.w; x++) {
            int componentId3 = segmented3.get_pixel(x, y) - 1;
            if (componentId3 != -1 && !keepComponent[componentId3]) {
                result.set_pixel(x, y, argb_to_col(0, 255, 255, 255));
            }
        }
    }

    return result;
}


vector<StepResult> find_intersections(const Pixels& p1, const Pixels& p2) {
    vector<StepResult> results;

    Pixels current_p1 = p1;
    Pixels current_p2 = p2;

    bool intersection_nonempty = 1;
    while (intersection_nonempty) {
        int max_x = 0;
        int max_y = 0;

        // Perform convolution mapping to find maximum intersection
        Pixels cm = convolve_map(current_p1, current_p2, max_x, max_y);

        // Intersect the two Pixels objects based on the maximum convolution
        // Pixels intersection = erase_small_components(intersect(current_p1, current_p2, max_x, max_y), 200);
        Pixels intersection = erase_low_iou(intersect(current_p1, current_p2, max_x, max_y), .6, current_p1, current_p2, max_x, max_y);

        intersection_nonempty = !intersection.is_empty();

        Pixels induced1 = induce(current_p1, intersection, 0, 0);
        Pixels induced2 = induce(current_p2, intersection, -max_x, -max_y);

        // Subtract the intersection from the starting Pixels
        current_p1 = remove_unconnected_components(subtract(current_p1, induced1, -current_p1.w, -current_p1.h));
        current_p2 = remove_unconnected_components(subtract(current_p2, induced2, -current_p2.w-max_x, -current_p2.h-max_y));

        // Store the result of this step
        StepResult step_result(max_x, max_y, cm, induced1, induced2, current_p1, current_p2, intersection);
        results.push_back(step_result);
    }

    return results;
}
