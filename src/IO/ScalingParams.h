enum class ScalingMode {
    BoundingBox,
    ScaleFactor
};

struct ScalingParams {
    ScalingMode mode;
    vec2 bounding_box;
    double scale_factor;

    // Constructors for different modes
    ScalingParams(vec2 bb)
        : mode(ScalingMode::BoundingBox), bounding_box(bb), scale_factor(0) {}

    ScalingParams(double factor) 
        : mode(ScalingMode::ScaleFactor), bounding_box(0, 0), scale_factor(factor) {}
};

