int convolve(const Pixels& a, const Pixels& b, int dx, int dy){
    /*a should be smaller for speed*/
    if(a.w*a.h>b.w*b.h) return convolve(b, a, -dx, -dy);

    double sum = 0;
    for (int x = 0; x < a.w; x++)
        for (int y = 0; y < a.h; y++){
            sum += a.get_alpha(x, y) > 0 && b.get_alpha(x-dx, y-dy) > 0;
        }
    return sum/4;
}

Pixels convolve_map(const Pixels& p1, const Pixels& p2, int& max_x, int& max_y){
    int max_conv = 0;
    Pixels ret(p1.w+p2.w, p1.h+p2.h);
    for(int x = 0; x < ret.w; x++)
        for(int y = 0; y < ret.h; y++){
            int convolution = convolve(p1, p2, x-p2.w, y-p2.h);
            if(convolution > max_conv){
                max_conv = convolution;
                max_x = x;
                max_y = y;
            }
            ret.set_pixel(x, y, makecol(convolution, 255, 255, 255));
        }
    max_x -= p2.w;
    max_y -= p2.h;
    return ret;
}

void flood_fill(Pixels& ret, const Pixels& p, int x, int y, int id) {
    // Check if current pixel is within bounds and has alpha channel != 0
    if (p.get_alpha(x, y) == 0)
        return;

    // Check if the current pixel is already segmented
    if (ret.get_alpha(x, y) != 0)
        return;

    // Set the identifier color for the current pixel
    ret.set_pixel(x, y, id);

    // Recursive flood fill for adjacent pixels
    flood_fill(ret, p, x + 1, y, id); // Right
    flood_fill(ret, p, x - 1, y, id); // Left
    flood_fill(ret, p, x, y + 1, id); // Down
    flood_fill(ret, p, x, y - 1, id); // Up
}


Pixels segment(const Pixels& p, int& id) {
    Pixels ret(p.w, p.h);

    // Initialize the identifier color
    id = 1;

    // Perform flood fill for each pixel in the input Pixels
    for (int y = 0; y < p.h; y++) {
        for (int x = 0; x < p.w; x++) {
            if (p.get_alpha(x, y) != 0 && ret.get_alpha(x, y) == 0) {
                // Found a new shape, perform flood fill starting from the current pixel
                flood_fill(ret, p, x, y, id);
                id++; // Increment the identifier color for the next shape
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
            if (p.get_alpha(x, y) == 255 && output.get_pixel(x, y) == 0) {
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

/*vector<Pixels> decompose(const Pixels& p) {
    // Segment the input Pixels
    int num_segments = 0;
    Pixels segmented_pixels = segment(p, num_segments);

    // Count the total number of segments
    int num_segments = 0;
    for (int y = 0; y < segmented_pixels.h; y++) {
        for (int x = 0; x < segmented_pixels.w; x++) {
            if (segmented_pixels.data[y][x].alpha > num_segments) {
                num_segments = segmented_pixels.data[y][x].alpha;
            }
        }
    }

    // Create a vector to store the decomposed segments
    vector<Pixels> ret(num_segments);

    // Find the bounding box for each segment
    vector<int> min_x(num_segments, INT_MAX);
    vector<int> min_y(num_segments, INT_MAX);
    vector<int> max_x(num_segments, INT_MIN);
    vector<int> max_y(num_segments, INT_MIN);

    for (int y = 0; y < segmented_pixels.h; y++) {
        for (int x = 0; x < segmented_pixels.w; x++) {
            int segment_id = segmented_pixels.data[y][x].alpha;
            if (segment_id > 0) {
                min_x[segment_id - 1] = min(min_x[segment_id - 1], x);
                min_y[segment_id - 1] = min(min_y[segment_id - 1], y);
                max_x[segment_id - 1] = max(max_x[segment_id - 1], x);
                max_y[segment_id - 1] = max(max_y[segment_id - 1], y);
            }
        }
    }

    // Crop and assign pixels to their corresponding segments
    for (int segment_id = 1; segment_id <= num_segments; segment_id++) {
        int width = max_x[segment_id - 1] - min_x[segment_id - 1] + 1;
        int height = max_y[segment_id - 1] - min_y[segment_id - 1] + 1;
        ret[segment_id - 1] = Pixels(width, height);
        
        for (int y = min_y[segment_id - 1]; y <= max_y[segment_id - 1]; y++) {
            for (int x = min_x[segment_id - 1]; x <= max_x[segment_id - 1]; x++) {
                if (segmented_pixels.data[y][x].alpha == segment_id) {
                    int relX = x - min_x[segment_id - 1];
                    int relY = y - min_y[segment_id - 1];
                    ret[segment_id - 1].data[relY][relX] = p.data[y][x];
                }
            }
        }
    }

    return ret;
}*/

Pixels intersect(const Pixels& p1, const Pixels& p2, int dx, int dy) {
    int width = p1.w;
    int height = p1.h;
    Pixels result(width, height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
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
    int width = p1.w;
    int height = p1.h;
    Pixels result(width, height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixel1 = p1.get_pixel(x, y);
            int pixel2 = p2.get_pixel(x - dx, y - dy);

            int alpha_diff = geta(pixel1) - geta(pixel2);
            int new_alpha = max(alpha_diff, 0);
            int new_pixel = makecol(new_alpha, getr(pixel1), getg(pixel1), getb(pixel1));

            result.set_pixel(x, y, new_pixel);
        }
    }

    return result;
}

struct StepResult {
    int max_x;
    int max_y;
    Pixels map;
    Pixels intersection;
    Pixels current_p1;
    Pixels current_p2;

    StepResult(int mx, int my, Pixels cm, Pixels intersect, Pixels p1, Pixels p2)
            : max_x(mx), max_y(my), map(cm), intersection(intersect), current_p1(p1), current_p2(p2) {}
};

vector<StepResult> find_intersections(const Pixels& p1, const Pixels& p2) {
    vector<StepResult> results;

    Pixels current_p1 = p1;
    Pixels current_p2 = p2;

    for (int i = 0; i < 4; i++) {
        int max_x = 0;
        int max_y = 0;

        // Perform convolution mapping to find maximum intersection
        Pixels cm = convolve_map(current_p1, current_p2, max_x, max_y);

        // Intersect the two Pixels objects based on the maximum convolution
        Pixels intersection = intersect(current_p1, current_p2, max_x, max_y);

        // Store the result of this step
        StepResult step_result(max_x, max_y, cm, intersection, current_p1, current_p2);
        results.push_back(step_result);

        // Subtract the intersection from the starting Pixels
        current_p1 = remove_unconnected_components(subtract(current_p1, intersection, 0, 0));
        current_p2 = remove_unconnected_components(subtract(current_p2, intersection, -max_x, -max_y));
    }

    return results;
}
