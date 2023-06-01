#pragma once

#include <cmath>

class Complex {
    public:
        double real, img;
        Complex(double r, double i) {
            real = r; img = i;
        }
        Complex operator +(Complex const &c) {
            return Complex(real+c.real, img+c.img);
        }
        Complex operator -(Complex const &c) {
            return Complex(real-c.real, img-c.img);
        }
        Complex operator *(Complex const &c) {
            return Complex(real*c.real-img*c.img, real*c.img+c.real*img);
        }
        double magnitude(){
            return std::hypot(real, img);
        }
        double magnitude2(){
            return real*real+img*img;
        }
};
