#include <cassert>

void unit_test_convolve() {
    // Create two Pixels objects with rectangular fills
    Pixels a(15, 15);
    Pixels b(10, 10);

    int col = argb_to_col(255,255,255,255);
    a.fill_rect(2, 2, 6, 6, col);  // Fill a rectangle in 'a'
    b.fill_rect(3, 3, 3, 3, col);  // Fill a smaller rectangle in 'b'

    // Test maximum overlap
    int max_overlap = convolve(a, b, 1, 1);
    int other_overlap = convolve(a, b, 5, 5);
    assert(max_overlap >= other_overlap);
    other_overlap = convolve(a, b, 1, 5);
    assert(max_overlap >= other_overlap);
    other_overlap = convolve(a, b, 1, 5);
    assert(max_overlap >= other_overlap);
}

void run_convolution_unit_tests(){
    //unit_test_convolve();
}
