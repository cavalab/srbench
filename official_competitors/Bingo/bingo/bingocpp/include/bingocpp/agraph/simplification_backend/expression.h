/*
 * Copyright 2018 United States Government as represented by the Administrator
 * of the National Aeronautics and Space Administration. No copyright is claimed
 * in the United States under Title 17, U.S. Code. All Other Rights Reserved.
 *
 * The Bingo Mini-app platform is licensed under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
*/
#ifndef INCLUDE_BINGOCPP_SIMPLIFICATION_BACKEND_EXPRESSION_H
#define INCLUDE_BINGOCPP_SIMPLIFICATION_BACKEND_EXPRESSION_H

#include <vector>
#include <iostream>
#include <typeinfo>
#include <memory>

#include <bingocpp/agraph/operator_definitions.h>

namespace bingo {
namespace simplification_backend {


class Expression: public std::enable_shared_from_this<Expression> {
  public:
    virtual ~Expression() = default;

    virtual Op GetOperator() const { return operator_; }

    virtual bool IsZero() const = 0;
    virtual bool IsOne() const = 0;
    virtual bool IsConstantValued() const = 0;
    //virtual std::vector<std::string> DependsOn() const = 0;
    virtual std::shared_ptr<const Expression> GetBase() const = 0;
    virtual std::shared_ptr<const Expression> GetExponent() const = 0;
    virtual std::shared_ptr<const Expression> GetTerm() const = 0;
    virtual std::shared_ptr<const Expression> GetCoefficient() const = 0;

    bool operator==(const Expression& other) const
    {
      if (typeid(*this) != typeid(other) || operator_ != other.operator_) return false;
      return equal(other);
    }
    inline bool operator!=(const Expression& other) const {
      return !(*this == other);
    }

    friend std::ostream &operator<<(std::ostream &strm, const Expression& expr)
      { return expr.print(strm); }

  protected:
    Op operator_;

    virtual bool equal(const Expression& other) const = 0;
    virtual std::ostream &print(std::ostream &strm) const = 0;
};


class TermExpression : public Expression {
  public:
    TermExpression(const Op operatr, const int operand);
    virtual ~TermExpression() = default;

    inline bool IsZero() const {
      return operator_ == kInteger && operand_ == 0;
    }
    inline bool IsOne() const {
      return operator_ == kInteger && operand_ == 1;
    }
    inline bool IsConstantValued() const {
      return operator_ != kVariable; // operator_ == kConstant || operator_ == kInteger;
    }

    //std::vector<std::string> DependsOn() const;
    inline std::shared_ptr<const Expression> GetBase() const { return shared_from_this(); }
    std::shared_ptr<const Expression> GetExponent() const;
    std::shared_ptr<const Expression> GetTerm() const;
    std::shared_ptr<const Expression> GetCoefficient() const;

  friend std::ostream &operator<<(std::ostream &strm, const TermExpression& expr)
      { return expr.print(strm); }

  private:
    int operand_;

    bool equal(const Expression& other) const;
    std::ostream &print(std::ostream &strm) const;
};


class OpExpression : public Expression {
  public:
    OpExpression(const Op operatr,
                 const std::vector<std::shared_ptr<const Expression>> operands);
    virtual ~OpExpression() = default;

    inline bool IsZero() const { return false; }
    inline bool IsOne() const { return false; }
    bool IsConstantValued() const;

    //std::vector<std::string> DependsOn() const;
    std::shared_ptr<const Expression> GetBase() const;
    std::shared_ptr<const Expression> GetExponent() const;
    std::shared_ptr<const Expression> GetTerm() const;
    std::shared_ptr<const Expression> GetCoefficient() const;

  friend std::ostream &operator<<(std::ostream &strm, const OpExpression& expr)
      { return expr.print(strm); }

  private:
    std::vector<std::shared_ptr<const Expression>> operands_;

    bool equal(const Expression& other) const;
    std::ostream &print(std::ostream &strm) const;
};



} // namespace simplification_backend
} // namespace bingo
#endif