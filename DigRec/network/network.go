package network
import (
    "math"
    "NeuralNetworks/DigRec/mnist"
    "NeuralNetworks/DigRec/mottuMat"
    "fmt"
)

type mottuNet struct {
    num_layers int
    sizes []int
    biases []*mottuMat.MottuMat
    weights []*mottuMat.MottuMat
}


/*
"""The list ``sizes`` contains the number of neurons in the
respective layers of the network.  For example, if the list
was [2, 3, 1] then it would be a three-layer network, with the
first layer containing 2 neurons, the second layer 3 neurons,
and the third layer 1 neuron.  The biases and weights for the
network are initialized randomly, using a Gaussian
distribution with mean 0, and variance 1.  Note that the first
layer is assumed to be an input layer, and by convention we
won't set any biases for those neurons, since biases are only
ever used in computing the outputs from later layers."""
*/
func MakeMottuNet(sizes []int) (*mottuNet) {

    retval := new(mottuNet)
    retval.num_layers = len(sizes)
    retval.sizes = sizes

    // In the example, the author creates this array as a using 
    // numpy.random.randn
    // Mottu: We need one vector of biases for each layer
    //        And we need one matrix of weights for each layer
    //        retval.weights[1= matrix W such that W[j][k] = weight for connection betwen 
    //        kth neuron in 2nd layer and jth neuron in the 3rd layer.
    //fmt.Println("------------------Biases--------------------")
    retval.biases = make([]*mottuMat.MottuMat, len(sizes)-1)
    for i := 0; i < len(retval.biases); i++ {
        retval.biases[i] = mottuMat.MakeMat(sizes[i+1], 1)
        retval.biases[i].Randomize()
        //fmt.Print("b", i, " = ")
        //retval.biases[i].Print()
    }

    //fmt.Println("------------------Weights--------------------") 
    retval.weights = make([]*mottuMat.MottuMat, len(sizes)-1)
    for i := 0; i < len(retval.weights); i++ {
        retval.weights[i] = mottuMat.MakeMat(sizes[i+1], sizes[i])
        retval.weights[i].Randomize()
        //fmt.Print("w", i, " = ")
        //retval.weights[i].Print()
    }
    return retval;
}

// Calclulate the sigmoid function
func sigmoid(z float64) float64 {
    return 1/(1+math.Exp(-z))
}

func sigmoid_prime(z float64) float64 {
    // d/dz of above function
    return sigmoid(z)*(1-sigmoid(z))
}

// Return the vector of partial C_x/partial a for the output activations
func cost_derivative(output_activations, y *mottuMat.MottuMat) *mottuMat.MottuMat{
    return output_activations.Sub(y)
}

// ==================== mottuNet functions ===============
func (this *mottuNet) FeedForward(a *mottuMat.MottuMat) *mottuMat.MottuMat {
    num_non_input_layers := this.num_layers-1  

    result := a
    for i := 0; i < num_non_input_layers; i++ {
        result = mottuMat.EvalLinMatExp(this.weights[i], result, this.biases[i])
        result.ApplyFuncEq(sigmoid)
    }
    return result
}


func (this *mottuNet) update_mini_batch(sw *mnist.Sweeper, mini_batch_size int, eta float64) {
    num_non_input_layers := this.num_layers-1
    nabla_b := make([]*mottuMat.MottuMat, num_non_input_layers)
    nabla_w := make([]*mottuMat.MottuMat, num_non_input_layers)

    
    x, y, present := sw.Next()
    if present {
        delta_nabla_b, delta_nabla_w := this.backprop(x, y)
        for i := 0; i < num_non_input_layers; i++ {
            nabla_b[i] = delta_nabla_b[i]
            nabla_w[i] = delta_nabla_w[i]
        }

        x, y, present = sw.Next()

        for present {
            delta_nabla_b, delta_nabla_w = this.backprop(x, y)
            for i := 0; i < num_non_input_layers; i++ {
                nabla_b[i].AddEq(delta_nabla_b[i])
                nabla_w[i].AddEq(delta_nabla_w[i])
            }

            x, y, present = sw.Next()
        }
    }
    factor := eta/float64(mini_batch_size)
    for i := 0; i < num_non_input_layers; i++ {
        nabla_w[i].ScaleEq(factor)
        this.weights[i].SubEq(nabla_w[i])

        nabla_b[i].ScaleEq(factor)
        this.biases[i].SubEq(nabla_b[i])
    }
}

func (this *mottuNet) backprop(x , y *mottuMat.MottuMat) ([]*mottuMat.MottuMat, []*mottuMat.MottuMat) {
    num_non_input_layers := this.num_layers-1
    nabla_b := make([]*mottuMat.MottuMat, num_non_input_layers)
    nabla_w := make([]*mottuMat.MottuMat, num_non_input_layers)

    // feedforward
    activation := x
    activations := make([]*mottuMat.MottuMat, this.num_layers) // num_non_input_layers+1
    activations[0] = activation
    zs := make([]*mottuMat.MottuMat, num_non_input_layers)
    for i := 0; i < num_non_input_layers; i++ {
        zs[i] = mottuMat.EvalLinMatExp(this.weights[i], activation, this.biases[i])
        activation = zs[i].ApplyFunc(sigmoid)
        activations[i+1] = activation
    }
    // backward pass
    

    delta := cost_derivative(activations[len(activations)-1], y)
    sp := zs[len(zs)-1].ApplyFunc(sigmoid_prime)
    delta.HadMulEq(sp)
    nabla_b[len(nabla_b)-1] = delta
    nabla_w[len(nabla_w)-1] = delta.BroadMul(activations[len(activations)-2])

    for l := 2; l < this.num_layers; l++ {

        sp = zs[len(zs)-l].ApplyFunc(sigmoid_prime)
        weights_t := this.weights[len(this.weights)-l+1].Transpose()
        delta = weights_t.Mul(delta)
        delta.HadMulEq(sp)

        nabla_b[len(nabla_b)-l] = delta
        nabla_w[len(nabla_w)-l] = delta.BroadMul(activations[len(activations)-l-1])
    }
    return nabla_b, nabla_w
}

/*
    Train the neural network using the mini-batch stochaistic
    gradient descent. The "training_data" is a struct of two 
*/
func (this *mottuNet) SGD(training_data *mnist.Set, epochs int, mini_batch_size int, eta float64) {
    n := training_data.Count()
    sw := training_data.Sweep()
 
    for j := 0; j < epochs; j++ {
        fmt.Println("Epoch ", j)
        sw.Shuffle()
        for k := 0; k <= n-mini_batch_size; k+= mini_batch_size {
            sw.SetBounds(k, k+mini_batch_size)
            this.update_mini_batch(sw, mini_batch_size, eta)
        }
    }
}

// Returns the number of test inputs for which mottuNet outputs the 
// correct result
func (this *mottuNet) Evaluate(test_data *mnist.Set) int {
    num_correct := 0
    sw := test_data.Sweep()
    image, exp_out, present := sw.Next()
    for present {
        test_result := this.FeedForward(image)
        count_to_add := 1
        max_act_index_mottu := -1
        max_val_mottu := 0.0
        max_act_index_exp := -1
        max_val_exp := 0.0
        result_len := test_result.Rows()
        
        for i := 0; i < result_len; i++ {
            if test_result.GetElem(i, 0) > max_val_mottu {
                max_act_index_mottu = i
                max_val_mottu = test_result.GetElem(i, 0)
            }

            if exp_out.GetElem(i, 0) > max_val_exp {
                max_act_index_exp = i
                max_val_exp = exp_out.GetElem(i, 0)
            }
        }
        if max_act_index_mottu != max_act_index_exp {
            count_to_add = 0
        }
        num_correct += count_to_add
        image, exp_out, present = sw.Next()
    }
    return num_correct
}
