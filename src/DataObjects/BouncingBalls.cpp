#pragma once

#include <vector>
#include "DataObject.cpp"

struct Ball {
    float x;
    float y;
    float vx;
    float vy;
};

class BouncingBalls : public DataObject {
public:
    vector<Ball> balls;
    int simulation_width;
    int simulation_height;
    float ball_radius;
    BouncingBalls(const int n, const int sw, const int sh, const float br) 
        : simulation_width(sw), simulation_height(sh), ball_radius(br) {
        for(int i = 0; i < n; i++) {
            float x = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - .5) * simulation_width;
            float y = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - .5) * simulation_height;
            float angle = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 2.0f * 3.14159265f;
            float vx = cos(angle) * br;
            float vy = sin(angle) * br;
            balls.push_back(Ball{x, y, vx, vy});
        }
        mark_updated();
    }

    void iterate() {

        for(size_t i = 0; i < balls.size(); i++) {
            for(size_t j = i + 1; j < balls.size(); j++) {
                float dx = balls[j].x - balls[i].x;
                float dy = balls[j].y - balls[i].y;
                float distance_squared = dx * dx + dy * dy;
                if(distance_squared < ball_radius * ball_radius * 4) {
                    // Compute perpendicular vector
                    float distance = sqrt(distance_squared);
                    float nx = dx / distance;
                    float ny = dy / distance;
                    // Compute relative velocity
                    float dvx = balls[j].vx - balls[i].vx;
                    float dvy = balls[j].vy - balls[i].vy;
                    // Compute velocity along the normal
                    float vn = dvx * nx + dvy * ny;
                    if(vn < 0) {
                        // Exchange velocities along the normal
                        balls[i].vx += vn * nx;
                        balls[i].vy += vn * ny;
                        balls[j].vx -= vn * nx;
                        balls[j].vy -= vn * ny;
                    }
                }
            }
        }

        for(auto& ball : balls) {
            if(ball.x <-simulation_width/2) { ball.vx = -ball.vx; ball.x =-simulation_width/2; }
            if(ball.x > simulation_width/2) { ball.vx = -ball.vx; ball.x = simulation_width/2; }
            
            if(ball.y <-simulation_height/2) { ball.vy = -ball.vy; ball.y =-simulation_height/2; }
            if(ball.y > simulation_height/2) { ball.vy = -ball.vy; ball.y = simulation_height/2; }
        }

        for(auto& ball : balls) {
            ball.x += ball.vx;
            ball.y += ball.vy;
        }

        mark_updated();
    }
};
