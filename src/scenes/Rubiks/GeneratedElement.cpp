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
    : PermutationElement(generate_construction   (elem1, elem2, method),
                         generate_effect         (elem1, elem2, method),
                         generate_primordial_size(elem1, elem2, method)) {
    }

    GeneratedElement(const shared_ptr<PermutationElement>& elem1,
                     int mult)
    : PermutationElement(to_string(mult) + elem1->get_name(),
                         mult==1?elem1->get_effect():generate_effect(elem1, make_shared<GeneratedElement>(elem1, mult-1), GenerationMethod::COMPOSITION),
                         elem1->get_primordial_size()*mult) {
    }

private:
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

    static int generate_primordial_size(const shared_ptr<PermutationElement>& elem1, 
                                        const shared_ptr<PermutationElement>& elem2, 
                                        GenerationMethod method) {
        switch (method) {
            case GenerationMethod::COMPOSITION: return elem1->get_primordial_size() + elem2->get_primordial_size();
            case GenerationMethod::CONJUGATION: return 2*elem1->get_primordial_size() + elem2->get_primordial_size();
            case GenerationMethod::COMMUTATION: return 2*elem1->get_primordial_size() + 2*elem2->get_primordial_size();
        }
        cout << "ERROR: was unable to generate element's primordial size." << endl;
        exit(1);
    }
};
