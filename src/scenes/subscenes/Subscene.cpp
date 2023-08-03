#pragma once

class SubsceneFactory;

class Subscene {
public:
    /**
     * Virtual destructor to ensure proper cleanup when deleting derived objects.
     */
    virtual ~Subscene() {}
    Subscene(int w, int h) : rendered(false), pixels(w, h) {}

    const Pixels& get() {
        if (!rendered) {
            render();
            rendered = true;
        }
        return pixels;
    }

    Pixels pixels;
    bool rendered;

protected:
    virtual void render() = 0;
};
