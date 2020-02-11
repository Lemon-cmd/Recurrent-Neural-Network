#include <string> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include <algorithm>
#include "MatrixCP.hpp"
#include "MatrixAdd.hpp"
#include "MatrixM.hpp"
#include "MatrixMult.hpp"

class RNN
{
    private:    
        MatrixMult mMult;
        MatrixAdd mAdd;
        MatrixCP mCP;
        MM mM;
        
        map <char, int> char_id;
        map <int, char> id_char;
        int hiddens;
        int sequence_length; 

        int data_size;
        int vocab_size;
        vector <vector <double> > W_XH;
        vector <vector <double> > W_HH;
        vector <vector <double> > W_HY;
        vector <vector <double> > B_H;
        vector <vector <double> > B_Y;
        vector <vector <double> > mW_XH;
        vector <vector <double> > mW_HH;
        vector <vector <double> > mW_HY;
        vector <vector <double> > mB_H;
        vector <vector <double> > mB_Y;
        

        void loss(const vector <int> &inputs, const vector <int> &targets, const vector <vector <double> > &hprev)
        {
            map <int, vector <vector <double> > > xs, hs, ys, ps;
            vector <vector <double> > original(hprev);
            hs[-1] = original;
            double loss = 0.0;

            for (int i = 0; i < inputs.size(); i ++)
            {
                xs[i] = generateB(vocab_size, 1, 0.0);
                mM.setItem(xs[i], inputs[i], 1);

                hs[i] = mM.mTanh
                (
                    mAdd.matrix_add(
                        mAdd.matrix_add(mCP.cross_product(W_XH, xs[i]), mCP.cross_product(W_HH, hs[i - 1])),
                        B_H
                    )
                );

                ys[i] = mAdd.matrix_add
                (
                    mCP.cross_product(W_HY, hs[i]),
                    B_Y
                );

                auto ys_exp = mM.mExp(ys[i]);

                ps[i] = mM.mDivide
                (
                    ys_exp, 
                    mM.mSum(ys_exp)
                );

                loss += -(log(ps[i][targets[i]][0]));

            }

            auto dWxh = zeros_like(W_XH);
            auto dWhy = zeros_like(W_HY);
            auto dWhh = zeros_like(W_HH);
            auto dbh = zeros_like(B_H);
            auto dby = zeros_like(B_Y);
            auto dh_next = zeros_like(hs[0]);

            for (int i = inputs.size() - 1; i >= 0; i --)
            {
                vector <vector <double> > dy (ps[i]);

                mM.addItem(dy, targets[i], -1);

                dWhy = mAdd.matrix_add
                (
                    dWhy,
                    mCP.cross_product(dy, mM.transpose(hs[i]))
                );

                dby = mAdd.matrix_add(dby, dy);

                
                auto dh = mAdd.matrix_add
                (
                    mCP.cross_product(mM.transpose(W_HY), dy),
                    dh_next
                );

                
                auto dhraw = mMult.matrix_mult
                (
                    mM.mSubtract(1, mMult.matrix_mult(hs[i], hs[i])),
                    dh
                );

                dbh = mAdd.matrix_add(dbh, dhraw);
                dWxh = mAdd.matrix_add(dWxh, mCP.cross_product(dhraw, mM.transpose(xs[i])));
                dWhh = mAdd.matrix_add(dWhh, mCP.cross_product(dhraw, mM.transpose(hs[i - 1])));
                dh_next = mCP.cross_product(mM.transpose(W_HH), dhraw);
                
            }

        }

        const string concat(const vector <int> samples)
        {
            string output = "";

            for (auto &id : samples)
            {
                output += id_char[id];
            }

            return output;
        }

        const vector <int> sample(const vector <vector <double> > &h_prev, const int &seedID, const int &m)
        {
            auto x = generateB(vocab_size, 1, 0.0);                             //generate a new bias called x
            mM.setItem(x, seedID, 1);                                           //set the entry at seedID equals to 1
            vector <int> samples;                                               //new samples of char IDs

            for (int i = 0; i < m; i ++)
            { 
                auto h = mM.mTanh
                (
                    mAdd.matrix_add
                    (
                        mAdd.matrix_add(mCP.cross_product(W_XH, x), mCP.cross_product(W_HH, h_prev)), 
                        B_H
                    ) 
                );

                auto y = mAdd.matrix_add(mCP.cross_product(W_HY, h), B_Y);
                auto exp_y = mM.mExp(y);
                auto p = mM.mDivide(exp_y, mM.mSum(exp_y)); 
                auto ix = rand() % (vocab_size - 1);
                x = generateB(vocab_size, 1, 0.0);
                mM.setItem(x, ix, 1);
                samples.push_back(ix);
            }

            return samples;

        }

        const vector <int> getID(const int &p, const int &extra)
        {
            /* Get a list of characters based on the range of p and sequence length */
            vector <int> item;
            int count = p;

            for (auto &key : char_id)
            {
                if (count < sequence_length + extra)
                {
                    item.push_back(key.second);
                    count ++;
                }
                else
                {
                    break;
                }
            }
            return item;
        }

        const vector < vector <double> > generateW (const int &row_size, const int &col_size, const double max_range)
        {
            /* Generate Random Weight Method */
            vector < vector <double> > weights;

            for (int r = 0; r < row_size; r ++)
            {
                vector <double> entry;
                for (int c = 0; c < col_size; c ++)
                {
                    //push back random weight onto current row
                    entry.push_back( (double(rand()) / double(RAND_MAX)) * max_range);
                }
                //push into the weights
                weights.push_back(entry);
            }

            return weights;
        }

        const vector <vector <double> > generateB (const int &row_size, const int &col_size, const double &bias_value)
        {
            /* Generate Biases */
            vector < vector <double> > biases;
            for (int r = 0; r < row_size; r ++)
            {
                //create entry and push into biases
                vector <double> entry;
                for (int c = 0; c < col_size; c ++)
                {
                    entry.push_back(bias_value);
                }

                biases.push_back(entry);
            }
            
            return biases;
        }

        const vector <vector <double> > zeros_like(const vector <vector <double> > &target)
        {
            /* Copy the dimension of target vectors and create a new vector with only zeroes */
            int row_size = target.size();
            int col_size = target[0].size();

            vector <vector <double> > zeroes;

            for (int r = 0; r < row_size; r ++)
            {
                vector <double> entry;
                for (int c = 0; c < col_size; c ++)
                {
                    entry.push_back(0.0);
                }

                zeroes.push_back(entry);
            }
            return zeroes;
        }

        void initalized()
        {
            W_XH = generateW(hiddens, vocab_size, 0.01);
            W_HH = generateW(hiddens, hiddens, 0.01);
            W_HY = generateW(vocab_size, hiddens, 0.01);
            B_H = generateB(hiddens, 1, 0.0);
            B_Y = generateB(vocab_size, 1, 0.0);

            mW_XH = zeros_like(W_XH);
            mW_HH = zeros_like(W_HH);
            mW_HY = zeros_like(W_HY);

            mB_H = zeros_like(B_H); 
            mB_Y = zeros_like(B_Y); 
        }

    public:   
        RNN(const int &hidden_size, const int &seq_length)
        {
            hiddens = hidden_size;
            sequence_length = seq_length;
            mMult = MatrixMult();
            mAdd = MatrixAdd();
            mCP = MatrixCP();
            mM = MM();
        }  
        
        void load(const string filename)
        {
            /* Load File and Enumerate through each character in the text file; get unique char and assign their ID */
            ifstream file(filename);    // grab file
            string line;                // a string for holding  current line
            int count = 0;              // count variable
            data_size = 0;
            vocab_size = 0;

            // loop through the file
            while (getline(file, line))
            {
                //if the current line is not empty
                if (!line.empty())
                {
                    //loop through each line and find character
                    for (int c = 0; c < line.size(); c ++)
                    {
                       if (char_id.find(line[c]) == char_id.end())
                       {
                            //assign each char a special ID
                            char_id[line[c]] = count;
                            id_char[count] = line[c];
                            count ++;
                            vocab_size ++;
                       }
                   }       
                }
                data_size ++;
            }
            initalized();
        }

        void learn()
        {            
            int n, p = 0;

            vector <vector <double> > h_prev;
            
            while (true)
            {
                if (p + sequence_length >= data_size or n == 0)
                {
                    h_prev = generateB(hiddens, 1, 0.0);
                    p = 0;
                }

                vector <int> inputs = getID(p, 0); 
                vector <int> targets = getID(p + 1, 1);

                if (n % 1000 == 0)
                {
                    auto samples = sample(h_prev, inputs[0], 200);
                    auto text = concat(samples);

                    cout << text << endl;
                }
            }
        }

        void display()
        {
            auto x = generateB(hiddens, 1, 5); 
            auto x2 = generateB(hiddens, 1, 5); 
            auto y = generateB(vocab_size, 1, 0.0);

            auto test = sample(x, 10, 200);
            int p = 0;

            vector <int> inputs = getID(0, 25); 
            vector <int> targets = getID(p + 1, 25);

            //cout << concat(test) << endl;
            loss(inputs, targets, x);
        }
};

int main()
{
    int hidden_size = 120;
    int seq_length = 25;
    double learning_rate = 1.0 * exp(-1);
    double smooth_loss = -log(1.0/100) * double(seq_length); 

    RNN test = RNN(hidden_size, seq_length);
    test.load("input.txt");
    test.display();

}