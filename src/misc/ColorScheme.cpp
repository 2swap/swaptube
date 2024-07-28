#pragma once

using namespace std;

#include <vector>
#include <list>
#include <sstream>
#include <iostream>

class ColorScheme {
private:
    list<unsigned int> colors;
public:
    ColorScheme(list<unsigned int> c) : colors(c) {}
    int get_color(){
        int color = colors.front();
        colors.pop_front();
        colors.push_back(color);
        return color;
    }
    ColorScheme(const string& hex_string) {
        for (size_t i = 0; i < hex_string.size(); i += 6) {
            string hex_color = hex_string.substr(i, 6);
            unsigned int x;   
            stringstream ss;
            ss << hex << hex_color;
            ss >> x;
            colors.push_back(x | 0xff000000);
        }
    }
};

vector<ColorScheme> get_color_schemes(){
    vector<ColorScheme> color_schemes;

    /* Thanks to Color Hunt (https://colorhunt.co/) for these sick schemes! */
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xff222831, 0xff393e46, 0xff00adb5, 0xffeeeeee}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xffe3fdfd, 0xffcbf1f5, 0xffa6e3e9, 0xff71c9ce}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xfff9f7f7, 0xffdbe2ef, 0xff3f72af, 0xff112d4e}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xffffc7c7, 0xffffe2e2, 0xfff6f6f6, 0xff8785a2}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xfffff5e4, 0xffffe3e1, 0xffffd1d1, 0xffff9494}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xffad8b73, 0xffceab93, 0xffe3caa5, 0xfffffbe9}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xff08d9d6, 0xff252a34, 0xffff2e63, 0xffeaeaea}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xfff4eeff, 0xffdcd6f7, 0xffa6b1e1, 0xff424874}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xfff9ed69, 0xfff08a5d, 0xffb83b5e, 0xff6a2c70}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xfff38181, 0xfffce38a, 0xffeaffd0, 0xff95e1d3}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xff1b262c, 0xff0f4c75, 0xff3282b8, 0xffbbe1fa}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xfff9f5f6, 0xfff8e8ee, 0xfffdcedf, 0xfff2bed1}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xffa8d8ea, 0xffaa96da, 0xfffcbad3, 0xffffffd2}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xffb7c4cf, 0xffeee3cb, 0xffd7c0ae, 0xff967e76}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xffb1b2ff, 0xffaac4ff, 0xffd2daff, 0xffeef1ff}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xffffb6b9, 0xfffae3d9, 0xffbbded6, 0xff61c0bf}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xff6096b4, 0xff93bfcf, 0xffbdcdd6, 0xffeee9da}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xffffeddb, 0xffedcdbb, 0xffe3b7a0, 0xffbf9270}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xff7d5a50, 0xffb4846c, 0xffe5b299, 0xfffcdec0}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xff364f6b, 0xff3fc1c9, 0xfff5f5f5, 0xfffc5185}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xff8d7b68, 0xffa4907c, 0xffc8b6a6, 0xfff1dec9}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xff27374d, 0xff526d82, 0xff9db2bf, 0xffdde6ed}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xffdefcf9, 0xffcadefc, 0xffc3bef0, 0xffcca8e9}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xfff8ede3, 0xffbdd2b6, 0xffa2b29f, 0xff798777}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xfffcd1d1, 0xffece2e1, 0xffd3e0dc, 0xffaee1e1}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xfffff8ea, 0xff9e7676, 0xff815b5b, 0xff594545}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xfff5efe6, 0xffe8dfca, 0xffaebdca, 0xff7895b2}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xffffe6e6, 0xfff2d1d1, 0xffdaeaf1, 0xffc6dce4}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xff2c3639, 0xff3f4e4f, 0xffa27b5c, 0xffdcd7c9}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xfff67280, 0xffc06c84, 0xff6c5b7b, 0xff355c7d}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xfffefcf3, 0xfff5ebe0, 0xfff0dbdb, 0xffdba39a}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xfffcd8d4, 0xfffdf6f0, 0xfff8e2cf, 0xfff5c6aa}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xffedf1d6, 0xff9dc08b, 0xff609966, 0xff40513b}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xff2b2e4a, 0xffe84545, 0xff903749, 0xff53354a}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xfff7fbfc, 0xffd6e6f2, 0xffb9d7ea, 0xff769fcd}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xffe4f9f5, 0xff30e3ca, 0xff11999e, 0xff40514e}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xff212121, 0xff323232, 0xff0d7377, 0xff14ffec}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xfff8ede3, 0xffdfd3c3, 0xffd0b8a8, 0xff7d6e83}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xff867070, 0xffd5b4b4, 0xffe4d0d0, 0xfff5ebeb}));
    color_schemes.push_back(ColorScheme(list<unsigned int>{0xfffdefef, 0xfff4dfd0, 0xffdad0c2, 0xffcdbba7}));
    return color_schemes;
}
