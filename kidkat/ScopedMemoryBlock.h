#ifndef SCOPEDMEMORYBLOCK_H_INCLUDED
#define SCOPEDMEMORYBLOCK_H_INCLUDED


template<typename T>
class ScopedMemoryBlock final
{
public:
    
    ScopedMemoryBlock() noexcept : data(nullptr), size(0) {}
    
    explicit ScopedMemoryBlock(const size_t numElements) :
    data(static_cast<T *>(std::calloc(numElements, sizeof(T)))),
    size(numElements)
    {
    }
    
    ~ScopedMemoryBlock()
    {
        this->clear();
    }
    
    inline T *getData() const noexcept                  { return this->data; }
    inline T& operator[] (size_t index) const noexcept  { return this->data[index]; }
    
    ScopedMemoryBlock &operator =(ScopedMemoryBlock &&other) noexcept
    {
        std::swap(this->data, other.data);
        this->size = other.size;
        return *this;
    }
    
    void clear() noexcept
    {
        std::free(this->data);
        this->data = nullptr;
        this->size = 0;
    }
    
    size_t getSize() const noexcept
    {
        return this->size;
    }
    
private:
    
    T *data;
    size_t size;
    
    TINYRNN_DISALLOW_COPY_AND_ASSIGN(ScopedMemoryBlock);
};


#endif  // SCOPEDMEMORYBLOCK_H_INCLUDED
