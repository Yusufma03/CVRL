在main的基础上增加state abstraction使得
如果z_{i+1}和z_{j+1}相同，那么r(s_i, a_i)和r(s_j, a_j)且a_i和a_j也相同，z_{i+1}~p(z_{i+1} \ z_{i}, a_{i}, o_{i+1})，z_{j+1}~p(z_{j+1} \ z_{j}, a_{j}, o_{j+1})，如果W(r(o_i, a_i), r(o_j, a_j))+W(a_i, a_j)小于epsilon=0.2，那么是1，否则为0
