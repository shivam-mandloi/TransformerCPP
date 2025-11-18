#pragma once
#include "vecX.hpp"

/*
    Used by the only Transformer Block
*/


vecX<double> AttentionQueryKeyMultiply(int h, vecX<double> &query, vecX<double> &key)
{
    // Multiply query and key with different head
    /*
        (Input):
            query: (h*d_q X s)
            key: (h*d_k X s)

            Here
                h -> number of heads
                d_q == d_k
        (Return):
            (h*s X s)
    */
    int qDim = query.row / h;
    query.TR(); // (s X h*d_q)
    vecX<double> res(h * query.row, query.row, 0);
    for(int head = 0; head < h; head++)
    {
        for(int seqRow = 0; seqRow < query.row; seqRow++)
        {
            for (int seqCol = 0; seqCol < query.row; seqCol++)
            {
                double temp = 0;
                for(int i = 0; i < qDim; i++)
                {
                    temp += (query.Get(seqRow, head * qDim + i) * key.Get(head * qDim + i, seqCol));
                }                
                res.push(head * query.row + seqRow, seqCol, temp);
            }
        }       
    }
    return res;
}

vecX<double> AttentionValueMultiply(int h, vecX<double> &attn, vecX<double> &value)
{
    /*
        (Input):
            attn: (h*s X s)
            value: (s X h*dv)

            Here
                h -> number of heads
                s -> sequence size
                dv -> Dimension of value
        (Return):
            (s X h*dv)
    */
    vecX<double> res(attn.col, value.col, 0);
    int dv = value.col / h;
    for(int head = 0; head < h; head++)
    {
        for(int seqRow = 0; seqRow < attn.col; seqRow++)
        {
            for(int vDim = 0; vDim < dv; vDim++)
            {
                for(int seqCol = 0; seqCol < attn.col; seqCol++)
                {
                    double temp = res.Get(head * attn.col + seqRow, vDim) + attn.Get(head * attn.col + seqRow, seqCol) * value.Get(seqCol, head * dv + vDim);
                    res.push(head * attn.col + seqRow, vDim, temp);
                }
            }
        }
    }
    return res;
}