#pragma once

#include <memory>
#include <string>
#include <utility>

#include "../../Core/Color.h"
#include "../../Core/Pixels.h"
#include "../DataObject.h"
#include "LambdaExpression.h"
#include "LambdaAbstraction.h"
#include "LambdaApplication.h"
#include "LambdaVariable.h"

using std::shared_ptr;
using std::weak_ptr;
using std::string;
using std::pair;

shared_ptr<LambdaExpression> apply(const shared_ptr<const LambdaExpression> f, const shared_ptr<const LambdaExpression> s, const int c, weak_ptr<LambdaExpression> p = shared_ptr<LambdaExpression>(), float x = 0, float y = 0, float w = 0, float h = 0, int u = 0);
shared_ptr<LambdaExpression> abstract(const char v, const shared_ptr<const LambdaExpression> b, const int c, weak_ptr<LambdaExpression> p = shared_ptr<LambdaExpression>(), float x = 0, float y = 0, float w = 0, float h = 0, int u = 0);

shared_ptr<LambdaExpression> get_interpolated_half(shared_ptr<LambdaExpression> l1, shared_ptr<const LambdaExpression> l2, const float weight);
pair<shared_ptr<LambdaExpression>, shared_ptr<LambdaExpression>> get_interpolated(shared_ptr<LambdaExpression> l1, shared_ptr<LambdaExpression> l2, const float weight);

shared_ptr<LambdaExpression> parse_lambda_from_string(const string& input);
