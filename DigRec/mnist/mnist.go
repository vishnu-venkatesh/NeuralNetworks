package mnist

/*
This code is from https://github.com/petar/GoMNIST 
It's purpose is to read the MNIST handwriting data.
*/

import (
    "compress/gzip"
    "encoding/binary"
    "image"
    "image/color"
    "io"
    "os"
    "fmt"
)

const (
    imageMagic = 0x00000803
    labelMagic = 0x00000801
    Width      = 28
    Height     = 28

)
// Image holds the pixel intensities of an image.
// 255 is the foreground (black), 0 is the background (white)
type RawImage []byte

func (img RawImage) ColorModel() color.Model {
    return color.GrayModel
}

func (img RawImage) Bounds() image.Rectangle {
    return image.Rectangle {
        Min: image.Point{0, 0},
        Max: image.Point{Width, Height},
    }
}

func (img RawImage) At(x, y int) color.Color {
    return color.Gray{img[y*Width+x]}
}

// ReadImageFile opens the named image file (training or test), parses it and
// returns all images in order.
func ReadImageFile(name string) (rows, cols int, imgs []RawImage, err error) {
    fmt.Println("ReadImageFile: Opening ", name);
    f, err := os.Open(name)
    if err != nil {
        fmt.Println("ReadImageFile: Could not open file ", name)
        return 0, 0, nil, err
    }
    defer f.Close()
    z, err := gzip.NewReader(f)
    if err != nil {
        fmt.Println("ReadImageFile: Issue with creating gzip reader")
        return 0, 0, nil, err
    }
    return readImageFile(z)
}

func readImageFile(r io.Reader) (rows, cols int, imgs []RawImage, err error) {
    var (
        magic int32
        n     int32
        nrow  int32
        ncol  int32
    )
    if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
        return 0, 0, nil, err
    }
    fmt.Printf("Read image magic num = %x\n", magic)
    if magic != imageMagic {
        return 0, 0, nil, os.ErrInvalid
    }
    if err = binary.Read(r, binary.BigEndian, &n); err != nil {
        return 0, 0, nil, err
    }
    fmt.Printf("Read num images = %d\n", n)
    if err = binary.Read(r, binary.BigEndian, &nrow); err != nil {
        return 0, 0, nil, err
    }
    fmt.Printf("Read num rows = %d\n", nrow) 
    if err = binary.Read(r, binary.BigEndian, &ncol); err != nil {
        return 0, 0, nil, err
    }
    fmt.Printf("Read num cols = %d\n", ncol)
    imgs = make([]RawImage, n)
    m := int(nrow * ncol)
    for i := 0; i < int(n); i++ {
        //fmt.Println("Trying to read image ", i)
        imgs[i] = make(RawImage, m)
        m_, err := io.ReadFull(r, imgs[i])
        if err != nil {
            return 0, 0, nil, err
        }
        if m_ != int(m) {
            return 0, 0, nil, os.ErrInvalid
        }
    }
    return int(nrow), int(ncol), imgs, nil
}

// Label is a digit label in 0 to 9
type Label uint8

//ReadLabelFile opens the named label file (training or test), parses it and
// returns all labels in order.
func ReadLabelFile(name string) (labels []Label, err error) {
    fmt.Println("Reading file ", name)
    f, err := os.Open(name)
    if err != nil {
        return nil, err
    }
    defer f.Close()
    z, err := gzip.NewReader(f)
    if err != nil {
        return nil, err
    }
    return readLabelFile(z)
}

func readLabelFile(r io.Reader) (labels []Label, err error) {
    var (
        magic int32
        n int32
    )
    if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
        return nil, err
    }
    fmt.Printf("Read magic number on label file = %x\n", magic)
    if magic != labelMagic {
        return nil, os.ErrInvalid
    }
    if err = binary.Read(r, binary.BigEndian, &n); err != nil {
        return nil, err
    }
    fmt.Printf("Read num labels = %d\n", n)
    labels = make([]Label, n)
    for i := 0; i < int(n); i++ {
        var l Label
        if err := binary.Read(r, binary.BigEndian, &l); err != nil {
            return nil, err
        }
        labels[i] = l
    }
    return labels, nil
}




/////////////// FOR DEBUG ONLY









