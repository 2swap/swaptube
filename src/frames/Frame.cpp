// Forward declaration of the Frame class
class Frame;

/**
 * Abstract base class for Frame objects representing a single frame of video data.
 */
class Frame {
public:
    /**
     * Virtual destructor to ensure proper cleanup when deleting derived objects.
     */
    virtual ~Frame() {}

    /**
     * Public function to get the rendered frame data.
     */
    const Pixels& get() {
        if (!rendered) {
            pixels = render();
            rendered = true;
        }
        return pixels;
    }

private:
    /**
     * Private virtual function to render the frame data.
     */
    virtual Pixels render() const = 0;

    // Private member variables
    Pixels pixels;
    bool rendered = false;
};
