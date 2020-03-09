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
            /* Structure for recording the result of the loss method */
            double loss;
            MatrixXd dWxh;
            MatrixXd dWhh;
            MatrixXd dWhy;
            MatrixXd dbh;
            MatrixXd dby;
            MatrixXd hs;   
        }; 

        vector <string> data;                                           //vector that holds the corpus
        map <string, int> char_id;                                      //map holds corpus unique char and their id
        map <int, string> id_char;                                      //map id to unique char

        int hiddens;                                                    //hidden layers number
        int sequence_length;                                            //sequence length
        int data_size;                                                  //length of corpus
        int vocab_size;                                                 //vocabulary size for the network

        MatrixXd W_XH;                                                  //Weights from Input to Hiddens
        MatrixXd W_HH;                                                  //Weights from Hidden to Hidden
        MatrixXd W_HY;                                                  //Weights from Hidden to Output
        MatrixXd BH;                                                    //Bias of Hiddens
        MatrixXd BY;                                                    //Bias of Outputs

        double check (const double min, const double max, double &target)
        {
            /* Sub-method to use for Clip Method */
            if (target > max)
            {
                return max;
            }
            else if (target < min)
            {
                return min;
            }
            return target;
        }

        void clip(const double min, const double max, MatrixXd &target)
        {
            /* Clip elements' value inside the Matrix to leave them remain in between min and max */
            for (int r = 0; r < target.rows(); r ++)
            {
                for (auto item = target.row(r).data(); item < target.row(r).data() + target.size(); item += target.outerStride()) 
                {
                    *item = check(min, max, *item);
                }   
            }
        }

        const vector <double> ravel(MatrixXd &target)
        {
            /* Generate a 1 by M Vector; Keep all values of target in 1D vector */
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
            /* Random Choice Method*/
            random_device rng;                                                  //set rng
            mt19937 gen(rng());                                                 //use another generator to generate upon rng
            boost::random::discrete_distribution<> dist (p.begin(), p.end());   //apply discrete distrib upon probabilities vector

            int randomNumber = dist(gen);                                       //set random ID & return
            return randomNumber;
        }

        loss_result* loss(const vector <int> &inputs, const vector <int> &targets, const MatrixXd &hprev)
        {
            /* Inputs and Targets are list of IDs 
               Hprev holds a matrix of initial Hidden States x 1
               Return Loss Result Structure
            */

            map <int, MatrixXd > xs, hs, ys, ps;     //map that holds iteration -> Matrix correspond to Inputs, Hiddens, Outputs, Probs
            MatrixXd original (hprev);               //Make a copy of hprev
            hs[-1] = original;                       //Set -1 as original; this is for the 1st iteration; hs[i - 1]
            double loss = 0.0;                       //Initialize loss

            //Forward Propagation
            for (int i = 0; i < inputs.size(); i ++)
            {
                xs[i] = MatrixXd::Zero(vocab_size, 1);             //Initalizes first X
                xs[i].row(inputs[i]).array() = 1;                  //encode X in 1 of K representation 
                
                //Apply Activation Function; Tanh(X * Weights of X->H + Weights of Hidden * Hidden States + Bias Hidden)
                hs[i] = ((W_XH * xs[i]) + (W_HH * hs[i - 1]) + BH).unaryExpr<double(*)(double)>(&tanh);
                
                //Set Outputs; Weights of Y * Hidden States + Bias Y
                ys[i] = (W_HY * hs[i]) + BY;
                
                //Set Exp(current Y)
                MatrixXd ys_exp = ys[i].unaryExpr<double(*)(double)>(&exp);

                //Set Probabilities of the next Y
                ps[i] = ys_exp / ys_exp.sum();

                //Set new loss 
                loss += -log(ps[i].coeff(targets[i], 0));
            }

            //Gradients
            MatrixXd dWxh = MatrixXd::Zero(W_XH.rows(), W_XH.cols());
            MatrixXd dWhy = MatrixXd::Zero(W_HY.rows(), W_HY.cols());
            MatrixXd dWhh = MatrixXd::Zero(W_HH.rows(), W_HH.cols());
            MatrixXd dbh = MatrixXd::Zero(BH.rows(), BH.cols());
            MatrixXd dby = MatrixXd::Zero(BY.rows(), BY.cols());
            MatrixXd dh_next = MatrixXd::Zero(hs[0].rows(), hs[0].cols());
            
            //Backpropagation: Computing Gradient 
            for (int i = inputs.size() - 1; i >= 0; i --)
            { 
                //copy current probabilities
                MatrixXd dy(ps[i]);    
                //back prop into y 
                dy.row(targets[i]) = dy.row(targets[i]).array() - 1.0;           
                
                //grab gradients of weights and biases of y
                dWhy += (dy * hs[i].transpose());
                dby.noalias() += dy;

                //back prop into h
                MatrixXd dh = ((W_HY.transpose() * dy) + dh_next);
                //back prop into tanh nonlinear
                MatrixXd dhraw = ((1 - (hs[i].array() * hs[i].array())).array() * dh.array()); 

                //set gradients
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
            
            /* Allocate necessary items into newly created Structure */
            loss_result* item = new loss_result();
            item->loss = loss; item->dbh = dbh; item->dby = dby;
            item->dWhh = dWhh; item->dWhy = dWhy; item->dWxh = dWxh;
            item->hs = hs[inputs.size() - 1];

            return item;
        }

        const vector <int> sample(const MatrixXd &h_prev, const int &seedID, const int &m)
        {
            /* Generate a Sample of Char ID for Text Generation */
            MatrixXd x = MatrixXd::Zero(vocab_size, 1);                      //generate a new bias called x
            x.row(seedID).array() = 1.0;                                     //set the entry at seedID equals to 1
            vector <int> samples;                                            //new samples of char IDs

            for (int i = 0; i < m; i ++)
            { 
                MatrixXd h = (W_XH * x) + (W_HH * h_prev) + BH;             //current Hypothesis
                h = h.unaryExpr<double(*)(double)>(&tanh);                  //apply tanh

                MatrixXd y = W_HY * h + BY;                                 //Produce Y
                MatrixXd exp_y = y.unaryExpr<double(*)(double)>(&exp);      //Apply Exp

                MatrixXd p = exp_y / exp_y.sum();                           //Create Probabilities 
                vector <double> probabilities = ravel(p);                   //Produce 1 by M vector

                int ix = choice(probabilities);                             //Select a choice based on probabilities

                x = MatrixXd::Zero(vocab_size, 1);                          //Reset X or Inputs

                x.row(ix).array() = 1.0;                                    //Set all elements in row SeedID equals to 1.0 again
                samples.push_back(ix);                                      //Push ID onto samples vector
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

        const string genText(const vector <int> samples)
        {
            /* Generate Text With Parallelism Enabled */
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

            /* Grab all lines in the text file */
            while(getline(file, line))
            {
                corpus.push_back(line);
            }

            file.close();               // close the file

            //set next line into the map and allocate the same ID into both map; then increment the ID
            char_id["\n"] = count; id_char[count] = "\n"; count ++; 

            //Loop through the corpus
            //Find Unique characters and allocate it with a unique ID
            //Also, append each character in the line onto the data vector for RNN's feeding
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

            data_size = data.size();                //set data size
            vocab_size = char_id.size();            //set vocab size
            
            /* Set Weights and Biases */
            W_XH = MatrixXd::Random(hiddens, vocab_size) * 0.01;
            W_HH = MatrixXd::Random(hiddens, hiddens) * 0.01;
            W_HY = MatrixXd::Random(vocab_size, hiddens) * 0.01;
            BH = MatrixXd::Zero(hiddens, 1);
            BY = MatrixXd::Zero(vocab_size, 1);
        }

        const void learn()
        {
            /* Learn Method */

            //Initalizes memory variables for Adam Gradient Descent aka AdaGrad
            MatrixXd mWXH = MatrixXd::Zero(W_XH.rows(), W_XH.cols());
            MatrixXd mWHH = MatrixXd::Zero(W_HH.rows(), W_HH.cols());
            MatrixXd mWHY = MatrixXd::Zero(W_HY.rows(), W_HY.cols());
            MatrixXd mBH = MatrixXd::Zero(BH.rows(), BH.cols());
            MatrixXd mBY = MatrixXd::Zero(BY.rows(), BY.cols());

            int n = 0;  //iteration
            int p = 0;  //incrementer for input & target

            double smooth_loss = -log(1.0/ double(vocab_size)) * double(sequence_length); 
            double learning_rate = 1 * exp(-1);
            
            MatrixXd h_prev; //saving previous hiddens

            while (true)
            {
                if ( ((p + sequence_length + 1) >= data_size) or (n == 0))
                {
                    h_prev = MatrixXd::Zero(hiddens, 1);  //Initalizes or Reset the memory of RNN
                    p = 0;                                //Set sequence incrementer back to 0
                }

                /* Grab Inputs and Targets */
                vector <int> inputs = getID(p, p + sequence_length); 
                vector <int> targets = getID(p + 1, p + sequence_length + 1);

                // Grab Loss Function's Outputs
                loss_result* item = loss(inputs, targets, h_prev);

                // Set new loss
                smooth_loss = smooth_loss * 0.999 + item->loss * 0.001;

                if ((n % 100 == 0) && (n != 0))
                {
                    /* Generate Text and Display Current Loss */
                    auto samples = sample(h_prev, inputs[0], 200);
                    auto text = genText(samples);

                    cout << "\n" << text << endl;
                    cout << "\n";
                    cout << "Iteration #: "  << n << " Loss: " << smooth_loss << endl;
                }
                
                if ((n % 10000 == 0) && (n != 0))
                {
                   // decreases learning rate as number of iteration increases
                   if (learning_rate >= 0.00000001)
                   {
                       learning_rate = learning_rate / (1.0 + (exp(-n/(n - 10000000))));
                   }
                   else
                   {
                       //restart
                       learning_rate = 0.001;
                   }
                       
                }

                /* Update Weight using AdaGrad*/
                mWXH.noalias() += MatrixXd(item->dWxh.array() * item->dWxh.array());
                W_XH += MatrixXd(-learning_rate * item->dWxh.array() / sqrt(mWXH.array() + 1e-8));

                mWHH.noalias() += MatrixXd(item->dWhh.array() * item->dWhh.array());
                W_HH += MatrixXd(-learning_rate * item->dWhh.array() / sqrt(mWHH.array() + 1e-8));
        
                mWHY.noalias() += MatrixXd(item->dWhy.array() * item->dWhy.array());
                W_HY += MatrixXd(-learning_rate * item->dWhy.array() / sqrt(mWHY.array() + 1e-8));

                /* Update Biases*/
                mBH.noalias() += MatrixXd(item->dbh.array() * item->dbh.array());
                BH += MatrixXd(-learning_rate * item->dbh.array() / sqrt(mBH.array() + 1e-8));

                mBY.noalias() += MatrixXd(item->dby.array() * item->dby.array());
                BY += MatrixXd(-learning_rate * item->dby.array() / sqrt(mBY.array() + 1e-8));

                delete(item);                           //free the loss result structure
                
                p += sequence_length;                   //update current sequence length
                n ++;                                   //increment current iteration
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
