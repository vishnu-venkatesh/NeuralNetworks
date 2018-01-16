package mottuMat

import (
    "math/rand"
//    "time"
    "fmt"
)

type MottuMat struct {
    data []float64
    numRows int
    numCols int
}

// MakeMat
func MakeMat(rows, cols int) *MottuMat {
    retval := new(MottuMat)
    retval.data = make([]float64, rows*cols)
    retval.numRows = rows
    retval.numCols = cols
    return retval
}


// MakeRowVec
func MakeRowVec(cols int) *MottuMat {
    return MakeMat(1, cols)
}
// MakeColVec
func MakeColVec(rows int) *MottuMat {
    return MakeMat(rows, 1)
}

// Evaluates the matrix expression: Ax+b
// A is a matrix (nxm)
// x is a col vec (mx1)
// b is a col vec (nx1)
//
func EvalLinMatExp(A, x, b *MottuMat) *MottuMat {
    if A.numCols != x.numRows || A.numRows != b.numRows || x.numCols != 1 || b.numCols != 1 {
        panic("Dimensions mismatch")
    }
    // nxm mxq
    n := A.numRows
    m := A.numCols
    result := MakeMat(n, 1)
    for i := 0; i < n; i++ {
        acc := b.data[i];
        for k := 0; k < m; k++ {
            acc += A.data[i*m+k]*x.data[k]
        }
        result.data[i] = acc 
    }
    return result
}


func (recv *MottuMat) Print() {
    if len(recv.data) == 0{
        return
    }
    fmt.Println("[")
    for i := 0; i < recv.numRows; i++ {
        fmt.Print("[", recv.data[i*recv.numCols])
        for j := 1; j < recv.numCols; j++ {
            fmt.Print(",", recv.data[i*recv.numCols+j]);
        }
        fmt.Print("],")
    }
    fmt.Println("]")
}


// Rows
func (recv *MottuMat) Rows() int {
    return recv.numRows
}
// Cols
func (recv *MottuMat) Cols() int {
    return recv.numCols
}

// Add
func (recv *MottuMat) Add(m *MottuMat) *MottuMat {
    if recv.numRows != m.numRows || recv.numCols != m.numCols {
        panic("Dimensions mismatch")
    }
    result := MakeMat(recv.numRows, recv.numCols)
    for i := 0; i < len(result.data); i++ {
        result.data[i] = recv.data[i] + m.data[i]
    }
    return result
}

func (recv *MottuMat) AddEq(m *MottuMat) {
    if recv.numRows != m.numRows || recv.numCols != m.numCols {
        panic("Dimensions mismatch")
    }
    for i := 0; i < len(recv.data); i++ {
        recv.data[i] += m.data[i]
    }
}
// Sub
func (recv *MottuMat) Sub(m *MottuMat) *MottuMat {
    if recv.numRows != m.numRows || recv.numCols != m.numCols {
        panic("Dimensions mismatch")
    }
    result := MakeMat(recv.numRows, recv.numCols)
    for i := 0; i < len(result.data); i++ {
        result.data[i] = recv.data[i] - m.data[i]
    }
    return result
}

func (recv *MottuMat) SubEq(m *MottuMat) {
    if recv.numRows != m.numRows || recv.numCols != m.numCols {
        panic("Dimensions mismatch")
    }
    for i := 0; i < len(recv.data); i++ {
        recv.data[i] -= m.data[i]
    }
}


// Mul
func (recv *MottuMat) Mul(B *MottuMat) *MottuMat {
    if recv.numCols != B.numRows {
        panic("Incompatible dimensions")
    }
    // nxm mxq
    n := recv.numRows
    m := recv.numCols
    q := B.numCols
    result := MakeMat(n, q)
    for i := 0; i < n; i++ {
        for j := 0; j < q; j++ {
            acc := 0.0;
            for k := 0; k < m; k++ {
                acc += recv.data[i*m+k]*B.data[k*q+j]
            }
            result.data[i*q+j] = acc
        }
    }
    return result
}


// Scale
func (recv *MottuMat) Scale(x float64) *MottuMat {
    result := MakeMat(recv.numRows, recv.numCols)
    for i := 0; i < len(recv.data); i++ {
        result.data[i] = x * recv.data[i]
    }
    return result
}

func (recv *MottuMat) ScaleEq(x float64) {
    for i := 0; i < len(recv.data); i++ {
        recv.data[i] *= x
    }
}

// ApplyFunc
func (recv *MottuMat) ApplyFunc(f func (float64) float64) *MottuMat {
    result := MakeMat(recv.numRows, recv.numCols)
    for i := 0; i < len(result.data); i++ {
        result.data[i] = f(recv.data[i])
    }
    return result
}

func (recv *MottuMat) ApplyFuncEq(f func (float64) float64) {
    for i := 0; i < len(recv.data); i++ {
        recv.data[i] = f(recv.data[i])
    }
}

// GetElem(i, j)
func (recv *MottuMat) GetElem(i, j int) float64 {
    return recv.data[i * recv.numCols + j]
}
// SetElem(i, j, val)
func (recv *MottuMat) SetElem(i, j int, val float64) {
    recv.data[i * recv.numCols + j] = val
}
// HadMul
func (recv *MottuMat)      HadMul(m *MottuMat) *MottuMat {
    if recv.numRows != m.numRows || recv.numCols != m.numCols {
        panic("Dimension mismatch")
    }
    result := MakeMat(recv.numRows, recv.numCols)
    for i := 0; i < len(result.data); i++ {
        result.data[i] = recv.data[i] * m.data[i]
    }
    return result
}

func (recv *MottuMat) HadMulEq(m *MottuMat) {
    if recv.numRows != m.numRows || recv.numCols != m.numCols {
        panic("Dimension mismatch")
    }
 
    for i := 0; i < len(recv.data); i++ {
        recv.data[i] *= m.data[i]
    }
}

// Transpose
func (recv *MottuMat) Transpose() *MottuMat {
    result := MakeMat(recv.numCols, recv.numRows)
    for i := 0; i < result.numRows; i++ {
        for j := 0; j < result.numCols; j++ {
            result.data[i * result.numCols + j ] = recv.data[j * recv.numCols + i]
        }
    }
    return result
}



// BroadMul
// Creates a matrix from two col vectors.
// The result[i][j] = recv[i] * m[j]
func (recv *MottuMat) BroadMul(m *MottuMat) *MottuMat {
    if !(recv.numCols == 1 && m.numCols == 1) {
        panic("Can't broadcast multiply.")
    }
    result := MakeMat(recv.numRows, m.numRows)
    for i := 0; i < result.numRows; i++ {
        for j := 0; j < result.numCols; j++ {
            result.data[i*result.numCols+j] = recv.data[i]*m.data[j]
        }
    }
    return result
}
// Randomize
func (recv *MottuMat) Randomize() {
    //time.Now().UnixNano() 
    r := rand.New( rand.NewSource(1))
    for i := 0; i < len(recv.data); i++ {
        recv.data[i] = r.NormFloat64()
    }
}
