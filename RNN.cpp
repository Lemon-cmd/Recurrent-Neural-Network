using namespace std;
#include <cassert>
#include <string> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <boost/random/discrete_distribution.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

class RNN
{
    private: 
        struct loss_result
        {
            double loss;
            MatrixXd dWxh;
            MatrixXd dWhh;
            MatrixXd dWhy;
            MatrixXd dbh;
            MatrixXd dby;
            MatrixXd hs;   
        }; 

        vector <string> data;
        vector <int> vocab_samples;

        int hiddens;
        int sequence_length;
        int data_size; 
        int vocab_size;


        MatrixXd mWXH;
        MatrixXd mWHH;
        MatrixXd mWHY;
        MatrixXd mBH;
        MatrixXd mBY;

        MatrixXd W_XH;
        MatrixXd W_HH;
        MatrixXd W_HY;
        MatrixXd BH; 
        MatrixXd BY;

        map <string, int> char_id;
        map <int, string> id_char;

        void setItem(MatrixXd &m, const int row_num, const double target)
        {
            for (auto item = m.row(row_num).data(); item< m.row(row_num).data() + m.size(); item += m.outerStride()) 
            {
                *item = target;
            }      
        }

        void clip(const double min, const double max, MatrixXd &target)
        {
            for (int r = 0; r < target.rows(); r ++)
            {
                for (auto item = target.row(r).data(); item < target.row(r).data() + target.size(); item += target.outerStride()) 
                {
                    *item = check(min, max, *item);
                }   
            }
        }

        void addItem(MatrixXd &m, const int row_num, const double target)
        {
            for (auto item = m.row(row_num).data(); item< m.row(row_num).data() + m.size(); item += m.outerStride()) 
            {
                *item += target;
            }   
        }

        double check (const double min, const double max, double &target)
        {
            if (target > max)
            {
                return max;
            }
            else if (target < min)
            {
                return min;
            }
            else 
            {
                return target;
            }
        }

        const vector <double> ravel(MatrixXd &target)
        {
            vector <double> p;
            
            for (int r = 0; r < target.rows(); r ++)
            {
                for (auto item = target.row(r).data(); item < target.row(r).data() + target.size(); item += target.outerStride())
                {
                    p.push_back(*item);
                }
            }

            return p;
        }

        const int choice (vector <double> &p)
        {
            random_device rng;
            mt19937 gen(rng());
            boost::random::discrete_distribution<> dist (p.begin(), p.end());

            int randomNumber = dist(gen);
            return randomNumber;
        }

        loss_result* loss(const vector <int> &inputs, const vector <int> &targets, const MatrixXd &hprev)
        {
            map <int, MatrixXd > xs, hs, ys, ps;
            MatrixXd original (hprev);
            hs[-1] = original;

            double loss = 0.0;
            for (int i = 0; i < inputs.size(); i ++)
            {
                xs[i] = MatrixXd::Zero(vocab_size, 1);
                xs[i].row(inputs[i]).array() = 1;

                hs[i] = ((W_XH * xs[i]) + (W_HH * hs[i - 1]) + BH).unaryExpr<double(*)(double)>(&tanh);
                
                ys[i] = (W_HY * hs[i]) + BY;

                MatrixXd ys_exp = ys[i].unaryExpr<double(*)(double)>(&exp);
            
                ps[i] = ys_exp / ys_exp.sum();

                loss += -log(ps[i].coeff(targets[i], 0));
            }

            MatrixXd dWxh = MatrixXd::Zero(W_XH.rows(), W_XH.cols());
            MatrixXd dWhy = MatrixXd::Zero(W_HY.rows(), W_HY.cols());
            MatrixXd dWhh = MatrixXd::Zero(W_HH.rows(), W_HH.cols());
            MatrixXd dbh = MatrixXd::Zero(BH.rows(), BH.cols());
            MatrixXd dby = MatrixXd::Zero(BY.rows(), BY.cols());
            MatrixXd dh_next = MatrixXd::Zero(hs[0].rows(), hs[0].cols());
            
            for (int i = inputs.size() - 1; i >= 0; i --)
            { 
                MatrixXd dy(ps[i]);
                dy.row(targets[i]) = dy.row(targets[i]).array() - 1.0;

                dWhy += (dy * hs[i].transpose());
                dby.noalias() += dy;

                MatrixXd dh = ((W_HY.transpose() * dy) + dh_next);
                MatrixXd dhraw = ((1 - (hs[i].array() * hs[i].array())).array() * dh.array()); 

                dbh.noalias() += dhraw;
                dWxh += (dhraw * xs[i].transpose());
                dWhh += (dhraw * hs[i - 1].transpose());
                dh_next = (W_HH.transpose() * dhraw);
            }
            
            /* Clip values in matrices to prevent gradient explosion */
            clip(-10, 10, dWxh);
            clip(-10, 10, dWhh);
            clip(-10, 10, dWhy);
            clip(-10, 10, dbh);
            clip(-10, 10, dby);
            
            loss_result* item = new loss_result();
            item->loss = loss; item->dbh = dbh; item->dby = dby;
            item->dWhh = dWhh; item->dWhy = dWhy; item->dWxh = dWxh;
            item->hs = hs[inputs.size() - 1];

            return item;
        }

        const vector <int> sample(const MatrixXd &h_prev, const int &seedID, const int &m)
        {
            MatrixXd x = MatrixXd::Zero(vocab_size, 1);                        //generate a new bias called x
            x.row(seedID).array() = 1.0;                                      //set the entry at seedID equals to 1
            vector <int> samples;                                            //new samples of char IDs

            for (int i = 0; i < m; i ++)
            { 
                MatrixXd h = (W_XH * x) + (W_HH * h_prev) + BH;
                h = h.unaryExpr<double(*)(double)>(&tanh);

                MatrixXd y = W_HY * h + BY;
                MatrixXd exp_y = y.unaryExpr<double(*)(double)>(&exp);

                MatrixXd p = exp_y / exp_y.sum();
                vector <double> probabilities = ravel(p);

                int ix = choice(probabilities);

                x = MatrixXd::Zero(vocab_size, 1);

                x.row(ix).array() = 1.0;
                samples.push_back(ix);
            }

            return samples;
        }
        
        const vector <int> getID(const int min, const int max)
        {
            // Get a list of characters based on the range of p and sequence length 
            vector <int> item;
      
            for (int c = min; c < max; c ++)
            {
                item.push_back(char_id[data[c]]);
            }
            
            return item;
        }

        const string concat(const vector <int> samples)
        {
            string output = "";
            #pragma omp declare reduction (merge : string : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
            #pragma omp parallel for reduction (merge : output)
            for (int i = 0; i < samples.size(); i++)
            {
                #pragma omp critical
                output += id_char[samples[i]];
            }

            return output;
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
            vector <string> corpus;

            while(getline(file, line))
            {
                corpus.push_back(line);
            }

            file.close();

            char_id["\n"] = count; id_char[count] = "\n"; count ++;

            for (auto &line: corpus)
            {
                int line_size = line.size();
                if (line_size > 0)
                {
                    for (int c = 0; c < line_size; c++)
                    {
                        string current (1, line[c]);
                        data.push_back(current);

                        if (char_id.find(current) == char_id.end())
                        {
                            char_id[current] = count;
                            id_char[count] = current;
                            count ++;
                        }
                    }

                    data.push_back("\n");
                }
                else
                {
                    data.push_back("\n");
                }
            }

            data_size = data.size();
            vocab_size = char_id.size();
            
            W_XH = MatrixXd::Random(hiddens, vocab_size) * 0.01;
            W_HH = MatrixXd::Random(hiddens, hiddens) * 0.01;
            W_HY = MatrixXd::Random(vocab_size, hiddens) * 0.01;
            BH = MatrixXd::Zero(hiddens, 1);
            BY = MatrixXd::Zero(vocab_size, 1);
        }

        const void learn()
        {

            mWXH = MatrixXd::Zero(W_XH.rows(), W_XH.cols());
            mWHH = MatrixXd::Zero(W_HH.rows(), W_HH.cols());
            mWHY = MatrixXd::Zero(W_HY.rows(), W_HY.cols());
            mBH = MatrixXd::Zero(BH.rows(), BH.cols());
            mBY = MatrixXd::Zero(BY.rows(), BY.cols());

            int n = 0;
            int p = 0;

            double smooth_loss = -log(1.0/ double(vocab_size)) * double(sequence_length); 
            double learning_rate = 1 * exp(-1);
            
            MatrixXd h_prev; 

            while (true)
            {
                if ( ((p + sequence_length + 1) >= data_size) or (n == 0))
                {
                    h_prev = MatrixXd::Zero(hiddens, 1);
                    p = 0;
                }

                vector <int> inputs = getID(p, p + sequence_length); 
                vector <int> targets = getID(p + 1, p + sequence_length + 1);

                if (n % 100 == 0)
                {
                    auto samples = sample(h_prev, inputs[0], 200);
                    auto text = concat(samples);

                    cout << "\n" << text << endl;
                    cout << "\n";
                }

                loss_result* item = loss(inputs, targets, h_prev);

                smooth_loss = smooth_loss * 0.999 + item->loss * 0.001;

                if (n % 100 == 0)
                {
                    cout << "Iteration #: "  << n << " Loss: " << smooth_loss << endl;
                }

                mWXH.noalias() += MatrixXd(item->dWxh.array() * item->dWxh.array());
                W_XH += MatrixXd(-learning_rate * item->dWxh.array() / sqrt(mWXH.array() + 1e-8));

                mWHH.noalias() += MatrixXd(item->dWhh.array() * item->dWhh.array());
                W_HH += MatrixXd(-learning_rate * item->dWhh.array() / sqrt(mWHH.array() + 1e-8));
        
                mWHY.noalias() += MatrixXd(item->dWhy.array() * item->dWhy.array());
                W_HY += MatrixXd(-learning_rate * item->dWhy.array() / sqrt(mWHY.array() + 1e-8));

                mBH.noalias() += MatrixXd(item->dbh.array() * item->dbh.array());
                BH += MatrixXd(-learning_rate * item->dbh.array() / sqrt(mBH.array() + 1e-8));

                mBY.noalias() += MatrixXd(item->dby.array() * item->dby.array());
                BY += MatrixXd(-learning_rate * item->dby.array() / sqrt(mBY.array() + 1e-8));

                delete(item);
                
                p += sequence_length;
                n ++;
            }
        }
};

int main()
{
    int hidden_size = 120;
    int seq_length = 25;

    Eigen::initParallel();
    RNN rnn = RNN(hidden_size, seq_length);
    rnn.load("input.txt");
    rnn.learn();
}

