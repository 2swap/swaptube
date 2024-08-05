#pragma once

#include <cmath>
#include <complex>

class Complex {
    public:
        double real, img;
        Complex() : real(0), img(0) {}
        Complex(double r, double i) : real(r), img(i) {}
        Complex operator +(double d) const {
            return Complex(real+d, img);
        }
        Complex operator -(double d) const {
            return Complex(real-d, img);
        }
        Complex operator *(double d) const {
            return Complex(d*real, d*img);
        }

        Complex operator +(const Complex &c) const {
            return Complex(real+c.real, img+c.img);
        }
        Complex operator -(const Complex &c) const {
            return Complex(real-c.real, img-c.img);
        }
        Complex operator *(const Complex &c) const {
            return Complex(real*c.real-img*c.img, real*c.img+c.real*img);
        }
        double magnitude() const {
            return std::hypot(real, img);
        }
        double magnitude2() const {
            return real*real+img*img;
        }
        Complex log() const {
            return Complex(std::log(magnitude()), atan2(img, real));
        }
        Complex exp() const {
            return Complex(std::exp(real) * cos(img), std::exp(real) * sin(img));
        }
        Complex pow(const Complex& exponent) const {
            /*double a = exponent.real;
            double b = exponent.img;

            double r = magnitude();
            double theta = std::atan2(img, real);

            double newR = std::pow(r, a) * std::exp(-b * theta);
            double newTheta = a * theta + 0.5 * b * std::log(r);

            return Complex(newR * cos(newTheta), newR * sin(newTheta));*/
            std::complex<double> z(real, img);
            std::complex<double> x(exponent.real, exponent.img);
            std::complex<double> a = std::pow(z, x);
            return Complex(a.real(), a.imag());
        }
        friend ostream& operator<<(std::ostream& os, const Complex& c) {
            os << c.real << " + " << c.img << "i";
            return os;
        }
};

bool operator==(const Complex& c1, const Complex& c2) {
    return (c1.real == c2.real) && (c1.img == c2.img);
}
bool operator!=(const Complex& c1, const Complex& c2) {
    return !(c1 == c2);
}
