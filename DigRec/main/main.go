package main

import (
    "os"
    "fmt"
    "NeuralNetworks/DigRec/mnist"
    "NeuralNetworks/DigRec/network"
)

func main() {
    args := os.Args

    if len(args) < 2 {
        fmt.Println("Usage: DigRec <Path To Dir with MNIST>")
        return
    }
    training_data, test_data, err := mnist.Load(args[1]) 
    if err != nil {
        fmt.Println("Encountered this error in loading MNIST data: ",
                    err.Error())
        return
    }
    sizes := []int{784, 30, 10}
    mn := network.MakeMottuNet(sizes)
    fmt.Println("MottuNet is studying ... really really hard :p")
    mn.SGD(training_data, 30, 10, 3.0)
    fmt.Println("Test time for mottu net ...")
    num_correct := mn.Evaluate(test_data)
    fmt.Println("How did mottu net do? %d out of %d correct",
                num_correct,
                test_data.Count())
    return

}
