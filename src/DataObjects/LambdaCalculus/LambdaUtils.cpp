#include "LambdaUtils.h"

#include <string>
#include <stdexcept>
#include <cassert>
#include <stack>
#include <memory>
#include <unordered_set>
#include <unordered_map>

#include "../../Core/Color.h"
#include "../../Core/Pixels.h"
#include "../DataObject.h"
#include "LambdaExpression.h"
#include "LambdaAbstraction.h"
#include "LambdaApplication.h"
#include "LambdaVariable.h"

using namespace std;

shared_ptr<LambdaExpression> apply(const shared_ptr<const LambdaExpression> f, const shared_ptr<const LambdaExpression> s, const int c, weak_ptr<LambdaExpression> p, float x, float y, float w, float h, int u){
    shared_ptr<LambdaExpression> nf = f->clone();
    shared_ptr<LambdaExpression> ns = s->clone();
    shared_ptr<LambdaExpression> ret = make_shared<LambdaApplication>(nf, ns, c, p, x, y, w, h, u);
    nf->set_parent(ret);
    ns->set_parent(ret);
    return ret;
}

shared_ptr<LambdaExpression> abstract(const char v, const shared_ptr<const LambdaExpression> b, const int c, weak_ptr<LambdaExpression> p, float x, float y, float w, float h, int u){
    shared_ptr<LambdaExpression> nb = b->clone();
    shared_ptr<LambdaExpression> ret = make_shared<LambdaAbstraction>(v, nb, c, p, x, y, w, h, u);
    nb->set_parent(ret);
    return ret;
}

void LambdaExpression::set_positions() {
    if(!parent.expired()) throw runtime_error("set_positions called on a child expression");

    int iter_x = 0;
    stack<shared_ptr<LambdaApplication>> applications_in_reverse;

    Iterator it = shared_from_this();
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        shared_ptr<const LambdaExpression> nearest_ancestor_application = current->get_nearest_ancestor_with_type("Application");
        current->x = iter_x+1;
        current->h = 2;
        if (current->get_type() == "Variable") {
            shared_ptr<const LambdaAbstraction> bound_abstr = dynamic_pointer_cast<LambdaVariable>(current)->get_bound_abstraction();
            current->y = bound_abstr->get_abstraction_depth() * 2;
            iter_x+=4;
        }
        else if(current->get_type() == "Abstraction") {
            current->y = current->get_abstraction_depth() * 2;
            current->w = current->num_variable_instantiations() * 4 - 1;
        }
        else if(current->get_type() == "Application") {
            shared_ptr<LambdaApplication> app = dynamic_pointer_cast<LambdaApplication>(current);
            current->w = app->get_first()->num_variable_instantiations() * 4 + 1;
            applications_in_reverse.push(app);
        }
    }

    while(!applications_in_reverse.empty()) {
        shared_ptr<LambdaApplication> current = applications_in_reverse.top();
        applications_in_reverse.pop();
        shared_ptr<const LambdaExpression> nearest_ancestor_abstraction = current->get_nearest_ancestor_with_type("Abstraction");
        current->y = max(nearest_ancestor_abstraction == nullptr ? 0 : nearest_ancestor_abstraction->y + 2, max(current->get_first()->get_height_recursive(), current->get_second()->get_height_recursive()));
    }

    it = shared_from_this();
    while (it.has_next()) {
        shared_ptr<LambdaExpression> current = it.next();
        shared_ptr<const LambdaExpression> nearest_ancestor = current->get_nearest_ancestor_with_type("Application");
        if (current->get_type() == "Variable") {
            if(nearest_ancestor == nullptr) {
                current->h = 2 * current->get_abstraction_depth() - current->y; // should this minus term be here?
            } else {
                current->h = nearest_ancestor->y - current->y + 1;
            }
        }
        else if(current->get_type() == "Application") {
            if(nearest_ancestor == nullptr) {
                current->h = 3;
            } else {
                current->h = nearest_ancestor->y - current->y + 1;
            }
        }
    }

    //TODO leave as much of this as possible uncomputed and rely on the draw function to compute it so that interpolation may look nicer?
}

Pixels LambdaExpression::draw_lambda_diagram(float scale = 1) {
    if(!parent.expired()) throw runtime_error("draw_lambda_diagram called on a child expression");
    if(w + h + x + y == 0) throw runtime_error("Attempted drawing lambda diagram with unset positions!");
    float bounding_box_w = get_width_recursive() + 4;
    float bounding_box_h = get_height_recursive() + 4;
    Pixels pix(bounding_box_w * scale, bounding_box_h * scale);

    for(int i = 0; i < 2; i++) {
        for(int step = 0; step < 2; step++) {
            Iterator it(shared_from_this());
            while (it.has_next()) {
                shared_ptr<LambdaExpression> current = it.next();
                int color = current->get_color();
                if((i==0)==(geta(color) == 255)) continue;
                if (current->get_type() == "Variable" && step == 0) {
                    pix.fill_rect((current->x+2) * scale, (current->y+2) * scale, scale, current->h * scale, color);
                }
                else if(current->get_type() == "Abstraction" && step == 1) {
                    pix.fill_rect((current->x+1) * scale, (current->y+2) * scale, current->w * scale, scale, color);
                }
                else if(current->get_type() == "Application" && step == 1) {
                    pix.fill_rect((current->x+2) * scale, (current->y+2) * scale, scale, current->h * scale, color);
                    pix.fill_rect((current->x+2) * scale, (current->y+2) * scale, current->w * scale, scale, color);
                }
            }
        }
    }
    return pix;
}

shared_ptr<LambdaExpression> get_interpolated_half(shared_ptr<LambdaExpression> l1, shared_ptr<const LambdaExpression> l2, const float weight){
    // Step 1: Create a map from uid to LambdaExpression pointers for l1
    unordered_map<int, shared_ptr<LambdaExpression>> l1_map;
    LambdaExpression::Iterator it_l1(l1);
    while (it_l1.has_next()) {
        shared_ptr<LambdaExpression> current_l1 = it_l1.next();
        l1_map[current_l1->get_uid()] = current_l1;
    }

    // Step 2: Iterate over it_ret and use the map for interpolation
    shared_ptr<LambdaExpression> ret = l2->clone();
    LambdaExpression::Iterator it_ret(ret);
    while (it_ret.has_next()) {
        shared_ptr<LambdaExpression> current_ret = it_ret.next();

        auto it = l1_map.find(current_ret->get_uid());
        if (it != l1_map.end()) {
            shared_ptr<LambdaExpression> current_l1 = it->second;
            current_ret->interpolate_positions(current_l1, weight);
            current_ret->set_color(colorlerp(current_ret->get_color(), current_l1->get_color(), weight));
        } else {
            current_ret->set_color(colorlerp(current_ret->get_color(), current_ret->get_color() & 0x00ffffff, weight));
        }
    }

    return ret;
}

pair<shared_ptr<LambdaExpression>, shared_ptr<LambdaExpression>> get_interpolated(shared_ptr<LambdaExpression> l1, shared_ptr<LambdaExpression> l2, const float weight){
    return make_pair(get_interpolated_half(l2, l1, weight), get_interpolated_half(l1, l2, 1-weight));
}

shared_ptr<const LambdaAbstraction> LambdaVariable::get_bound_abstraction() const {
    shared_ptr<const LambdaExpression> current = parent.lock();
    while (current) {
        if (current->get_type() == "Abstraction") {
            shared_ptr<const LambdaAbstraction> abstraction = dynamic_pointer_cast<const LambdaAbstraction>(current);
            if (abstraction && abstraction->get_bound_variable() == varname) {
                return abstraction;
            }
        }
        current = current->get_parent();
    }
    return nullptr;
}

shared_ptr<LambdaExpression> parse_lambda_from_string(const string& input) {
    if (is_single_letter(input))
        return make_shared<LambdaVariable>(input[0], OPAQUE_WHITE);

    assert(input.size() > 3);

    if(input[0             ] == '('  &&
       input[1             ] == '\\' &&
       input[3             ] == '.'  &&
       input[4             ] == ' '  &&
       input[input.size()-1] == ')'){
        shared_ptr<LambdaExpression> b = parse_lambda_from_string(input.substr(5, input.size() - 6));
        return abstract(input[2], b, OPAQUE_WHITE);
    }

    // Parsing application
    if (input[0] == '(' && input[input.size() - 1] == ')') {
        int level = 0;
        for (size_t i = 1; i < input.size() - 1; ++i) {
            if (input[i] == '(') level++;
            if (input[i] == ')') level--;
            if (input[i] == ' ' && level == 0) {
                shared_ptr<LambdaExpression> f = parse_lambda_from_string(input.substr(1, i - 1));
                shared_ptr<LambdaExpression> s = parse_lambda_from_string(input.substr(i + 1, input.size() - i - 2));
                return apply(f, s, OPAQUE_WHITE);
            }
        }
    }

    // No valid pattern was matched
    throw runtime_error("Failed to parse string!");
    return nullptr;
}
