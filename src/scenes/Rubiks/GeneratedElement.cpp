#pragma once

#include "PermutationElement.cpp"
#include <memory>
#include <algorithm>

enum class GenerationMethod { COMPOSITION, CONJUGATION, COMMUTATION };

class GeneratedElement : public PermutationElement {
public:
    GeneratedElement(const shared_ptr<PermutationElement>& elem1,
                     const shared_ptr<PermutationElement>& elem2,
                     GenerationMethod method)
    : PermutationElement(generate_construction(elem1, elem2, method), generate_effect(elem1, elem2, method)) {
    }

    GeneratedElement(const shared_ptr<PermutationElement>& elem1,
                     int mult)
    : PermutationElement(to_string(mult) + elem1->get_name(), mult==1?elem1->get_effect():generate_effect(elem1, make_shared<GeneratedElement>(elem1, mult-1), GenerationMethod::COMPOSITION)) {
    }

    void print() const override {
        cout << "GeneratedElement '" << name << "' modifies " + to_string(get_modified_set().size()) + " pieces." << endl;
    }

private:
    static vector<int> generate_effect(const shared_ptr<PermutationElement>& elem1, 
                                            const shared_ptr<PermutationElement>& elem2, 
                                            GenerationMethod method) {
        switch (method) {
            case GenerationMethod::COMPOSITION: return elem1->get_effect() + elem2->get_effect();
            case GenerationMethod::CONJUGATION: return elem1->get_effect() + elem2->get_effect() + ~(elem1->get_effect());
            case GenerationMethod::COMMUTATION: return elem1->get_effect() + elem2->get_effect() + ~(elem1->get_effect()) + ~(elem2->get_effect());
        }
        cout << "ERROR: was unable to generate element's effect." << endl;
        exit(1);
    }

    static string generate_construction(const shared_ptr<PermutationElement>& elem1, 
                                             const shared_ptr<PermutationElement>& elem2, 
                                             GenerationMethod method) {
        switch (method) {
            case GenerationMethod::COMPOSITION: return       elem1->get_name() + "+" + elem2->get_name();
            case GenerationMethod::CONJUGATION: return "<" + elem1->get_name() + "," + elem2->get_name() + ">";
            case GenerationMethod::COMMUTATION: return "[" + elem1->get_name() + "," + elem2->get_name() + "]";
        }
        cout << "ERROR: was unable to generate element's construction." << endl;
        exit(1);
    }

    static inline bool invalid_char(char c) {
        return !(isdigit(c) || islower(c));
    }

    bool string_is_valid(const string &str) {
        for(int i = 0; i < str.size(); i++){
            if(invalid_char(str[i]))
                return false;
        }
        return true;
    }
};
