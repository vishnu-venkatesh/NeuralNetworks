package mnist
/*
    Code from: https://github.com/petar/GoMNIST/blob/master/util.go
*/
import (
    "path"
    "math/rand"
    "NeuralNetworks/DigRec/mottuMat"
//    "time"
)

const NUM_TYPES_OF_DIGITS int = 10

// Set represents a data set of image-label pairs held in memory
type Set struct {
    NRow int
    NCol int
    Images []*mottuMat.MottuMat // Each element is the image flattened in row major order (matrix of nx1)
    ExpOut  []*mottuMat.MottuMat // Expected outputs. Each element is a matrix of 10x1

}

// ReadSet reads a set from the images file iname and the the corresponding
// labels file lname
func ReadSet(iname, lname string) (set *Set, err error) {
    set = &Set{}
    var rows, cols int
    var raw_images []RawImage
    var labels []Label

    if rows, cols, raw_images, err = ReadImageFile(iname); err != nil {
        return nil, err
    }
    if labels, err = ReadLabelFile(lname); err != nil {
        return nil, err
    }
    set.NRow = rows
    set.NCol = cols
    set.Images = make([]*mottuMat.MottuMat, len(raw_images))
    for i := 0; i < len(set.Images); i++ {
        nelems := rows * cols
        set.Images[i] = mottuMat.MakeMat(nelems, 1) 
        for j := 0; j < nelems; j++ {
            set.Images[i].SetElem(j, 0, float64(raw_images[i][j])/255.0)
        }
    }
    set.ExpOut = make([]*mottuMat.MottuMat, len(labels))
    for i := 0; i < len(labels); i++ {
        set.ExpOut[i] = mottuMat.MakeMat(NUM_TYPES_OF_DIGITS, 1)
        for j := 0; j < NUM_TYPES_OF_DIGITS; j++ {
            if labels[i] == Label(j) {
                set.ExpOut[i].SetElem(j, 0, 1.0)
            } else {
                set.ExpOut[i].SetElem(j, 0, 0.0)
            }
        }
    }

    return
}

// Count returns the number of points available in the data set
func (s *Set) Count() int {
    return len(s.Images)
}

// Get rturns the ith image and its corresponding label
func (s *Set) Get(i int) (*mottuMat.MottuMat, *mottuMat.MottuMat) {
    return s.Images[i], s.ExpOut[i]
}


// Sweeper is an iterator over the points in a data set
type Sweeper struct {
    set *Set
    i int
    upper_bound int // can't exceed length of set
    access_indices []int // indices used to access the set
    r *rand.Rand
}

// Next returns the next image and its label in the data set
// If the end is reached, present is set to false
func (sw *Sweeper) Next() (image, expOut *mottuMat.MottuMat, present bool) {
    if sw.i >= sw.upper_bound || sw.i >= len(sw.set.Images) {
        return image, expOut, false
    }
    // Mottu: i in sw is never incremented in the original. I'm fixing that here.
    sw.i++
    return sw.set.Images[sw.access_indices[sw.i-1]], sw.set.ExpOut[sw.access_indices[sw.i-1]], true
}

// Sets the bounds of the sweeper
func (sw *Sweeper) SetBounds(begin, end int) {
    sw.i = begin
    sw.upper_bound = end
}

// Shuffles the order in which the underlying set is accessed
// Resets the bounds to the entire set
func (sw *Sweeper) Shuffle() {
    sw.i = 0
    sw.upper_bound = sw.set.Count()
    sw.access_indices = sw.r.Perm(sw.set.Count())
}

// Sweep creates a new sweep iterator over the data set
func (s *Set) Sweep() *Sweeper {
    sw := new(Sweeper)
    sw.set = s
    sw.i = 0
    sw.upper_bound = s.Count()
    sw.r = rand.New( rand.NewSource(1))
    sw.access_indices = make([]int, s.Count())
    for i := 0; i < s.Count(); i++ {
        sw.access_indices[i] = i;
    }
    return sw
}


// Load reads both the training and the testing MNIST data sets, given
// a local directory dir, containing the MNIST disribution files
func Load(dir string) (train, test *Set, err error) {
    tr_im_str := "train-images-idx3-ubyte.gz"
    tr_lab_str := "train-labels-idx1-ubyte.gz"
    t10k_im_str := "t10k-images-idx3-ubyte.gz"
    t10k_lab_str := "t10k-labels-idx1-ubyte.gz"
    if train, err = ReadSet(path.Join(dir, tr_im_str), path.Join(dir, tr_lab_str)); err != nil {
        return nil, nil, err
    }
    if test, err = ReadSet(path.Join(dir, t10k_im_str), path.Join(dir, t10k_lab_str)); err != nil {
        return nil, nil, err
    }
    return
}
