#include <string> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include <vector>
#include <algorithm>

using namespace std;

class RNN
{
    private:    
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
            
        const vector <double> ravel (const vector <vector <double> > &v1)
        {
            assert(v1.size() > 0 && "Vector is empty!");

            vector <double> item;
            for (auto &row: v1)
            {
                for (auto &col: row)
                {
                    item.push_back(col);
                }
            }

            return item;
        }

        const vector <vector <double> > mDivide(vector <vector <double> > &v1, const double &target)
        {
            assert(v1.size() > 0 && "Vector is empty!");
            vector <vector <double> > item;
            for (auto &row : v1)
            {
                vector <double> elements;
                for (auto &col: row)
                {
                    elements.push_back(col / target);
                }
                item.push_back(elements);
            }

            return item;
        }

        const double mSum(const vector <vector <double> > &v1)
        {
            assert(v1.size() > 0 && "Vector is empty!");
            double sum = 0.0;

            for (auto &row: v1)
            {
                for (auto &col: row)
                {
                    sum += col;
                }
            }

            return sum;
        }
        
        const vector < vector <double> > mExp(const vector <vector <double> > &v1)
        {
            /* Perform exp(x) on Matrix */
            assert(v1.size() > 0 && "Vector is empty!");
            vector <vector <double> > item;

            for (int r = 0; r < v1.size(); r ++)
            {
                vector <double> elements;
                for (int c = 0; c < v1[0].size(); c ++)
                {
                    elements.push_back(exp(v1[r][c]));
                }
                item.push_back(elements);
            }

            return item;
        }

        const vector < vector <double> > mTanh(const vector <vector <double> > &v1)
        {
            /* Perform Tan(x) on Matrix */
            assert(v1.size() > 0 && "Vector is empty!");
            vector <vector <double> > item;

            for (int r = 0; r < v1.size(); r ++)
            {
                vector <double> elements;
                for (int c = 0; c < v1[0].size(); c ++)
                {
                    elements.push_back(tan(v1[r][c]));
                }
                item.push_back(elements);
            }

            return item;

        }

        const vector < vector <double> > mAdd(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            /* Matrices Addition */
            assert(v1.size() == v2.size() && v1[0].size() == v2[0].size() && "The size of the vectors do not match!");
            vector <vector <double> > item;

            for (int r = 0; r < v1.size(); r ++)
            {
                vector <double> elements;
                for (int c = 0; c < v1[0].size(); c ++)
                {
                    elements.push_back(v1[r][c] + v2[r][c]);
                }
                item.push_back(elements);
            }

            return item;

        }

        const vector < vector <double> > dot(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            /* Perform Matrices Multiplication; Dot Product */
            assert (v1[0].size() == v2.size() && "Vector 1's column size does not match Vector 2's row size!!!");
            vector <vector <double> > item;
            vector <double> elements;
            
            //start with the row in v1
            for (auto &row : v1)
            {
                // loop through each item in row
                for (int c = 0; c < row.size(); c ++)
                {
                    if (elements.size() > 0)
                    {
                        //clear elements it has elements; this is due to declaring elements outside of the loop
                        elements.clear();
                    }
                    //sum
                    double sum = 0.0;
                    //loop through the column of v2
                    for (int v2_c = 0; v2_c < v2[0].size(); v2_c ++)
                    {
                        //loop through row
                        for (auto &v2_row: v2)
                        {
                            //calculate sum 
                            sum += row[c] * v2_row[v2_c];
                        }
                        //push into elements vector
                        elements.push_back(sum);
                    }
                }
                //add new entry into item; new row
                item.push_back(elements);
            }
            return item;
        }
        
        void setItem(vector <vector <double> > &v1, const int &rowID, const double &target)
        {
            for (auto &col : v1[rowID])
            {
                col = target;
            }
        }

        const vector <int> sample(const vector <vector <double> > &h_prev, const int &seedID, const int &m)
        {
            auto x = generateB(vocab_size, 1, 0.0);                             //generate a new bias called x
            setItem(x, seedID, 1);                                              //set the entry at seedID equals to 1
            vector <int> samples;                                               //new samples of char IDs

            for (int i = 0; i < m; i ++)
            {
                auto h = mTanh(
                    mAdd(
                        mAdd(dot(W_XH, x), dot(W_HH, h_prev)), 
                        B_H
                        ) 
                );

                auto y = mAdd(dot(W_HY, h), B_Y);
                auto exp_y = mExp(y);
                auto p = mDivide(exp_y, mSum(exp_y)); 
                auto ix = rand() % (vocab_size - 1);
                x = generateB(vocab_size, 1, 0.0);
                setItem(x, ix, 1);

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
                    continue;
                }

            }
        }

        void display()
        {
            //vector < vector <double> > x = generateB(hiddens, 1, 0.0);  
           
            auto test = sample(B_H, 1, 100);
            for (auto &col : test)
            {
                cout << col << endl;
            }
            
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