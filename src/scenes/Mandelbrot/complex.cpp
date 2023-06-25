#pragma once

#include <cmath>

class Complex {
    public:
        double real, img;
        Complex() : real(0), img(0) {}
        Complex(double r, double i) : real(r), img(i) {}
        Complex operator +(const Complex &c) const {
            return Complex(real+c.real, img+c.img);
        }
        Complex operator -(const Complex &c) const {
            return Complex(real-c.real, img-c.img);
        }
        Complex operator *(const Complex &c) const {
            return Complex(real*c.real-img*c.img, real*c.img+c.real*img);
        }
        /*double magnitude(){
            return std::hypot(real, img);
        }*/
        double magnitude2() const {
            return real*real+img*img;
        }
        friend ostream& operator<<(std::ostream& os, const Complex& c) {
            os << c.real << " + " << c.img << "i";
            return os;
        }
};
