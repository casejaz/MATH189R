## ChunkyString Pseudocode

### `ChunkyString::Iterator` Operations

#### `operator++()`

```
    Throw error if the operator == end().
    
    if (iterator == end of the last chunk/ last character in the chunk){
        change it to end()
    } else if (iterator == end of any other chunk/ last character in the chunk){
        move it to the beginning of the next chunk
    } else {
        move it to the next character in the chunk
    }
```

#### `operator==(rhs)`

```
    if (both iterators are end()){
        return True;
    } else if (all of the data members of the two iterators are the same){
        return True;
    } else {
        return False;
    }
```


### `ChunkyString` Operations

#### `operator+=(rhs)`

```
    for (iterator from rhs.begin() to rhs.end()){
        call push_back(*iterator/ a character) to add it to the back of 
        the original list
    }

    // This for loop does nothing if rhs is an empty string. 
```


#### `iterator begin()`

```
    if (the list is an empty list) {
        return an invalid iterator, i.e. end()
    } else {
        return an iterator pointing to the first character in the first chunk
    }
```


#### `iterator end()`

```
    Always return an invalid iterator. 
```

#### `push_back(char c)`

```
    if (string == empty string){
        create a new chunk on the heap;
        add the address to the empty list of chunks 
        put char c to the first spot in the character array of the new chunk;
        increment the size of the new chunk;
        increment the size of the string;
    } else if (the last chunk of the string is full) {
        create a new chunk on the heap;
        store the address to the back of the existing list of chunks 
        put char c to the first spot in the character array of the new chunk;
        increment the size of the new chunk;
        increment the size of the string;
    } else {
        find the last chunk;
        put char c to the next available spot in the character array; 
        increment the size of the last chunk;
        increment the size of the string;
    }
```


#### `insert(char c, iterator i)`

```
    if (iterator == end()){
        call push_back(char c)
        return an iterator pointing to the last character of the last chunk
    } else if (iterator points to the first character of a chunk) {
        if (the chunk is not full){
            move all existing characters in the array one spot below;
            put char c to the first spot;
            increment the size of the chunk;
            increment the size of the string;
            return the original iterator;
        } else if (the chunk is the first chunk, i.e. iterator== begin()) {
            create a new chunk on the heap;
            add the address to the front of the existing list of chunks; 
            put char c to the first spot in the character array of the new chunk;
            increment the size of the new chunk;
            increment the size of the string;
            return begin()
        } else if (the previous chunk is not full) {
            put char c to the next available spot in the character array 
                of the previous chunk;
            increment the size of the previous chunk;
            increment the size of the string;
            return --original iterator;
        } else, i.e. both the current and previous chunk are full {
            create a new chunk on the heap;
            add the address to the front of the current list of chunks, 
                i.e. between the current and previous chunk;
            put char c to the first spot in the character array of the new chunk;
            increment the size of the new chunk;
            increment the size of the string;
            return --original iterator;
        }
    } else, i.e. the iterator points to the middle or the back of the chunk {
        if (the chunk is not full){
            move all characters after the iterator one spot below;
            put char c to the current spot;
            increment the size of the chunk;
            increment the size of the string;
            return the original iterator;
        } else if (the chunk is the first chunk and it is full) {
            create a new chunk on the heap;
            add the address to the front of the existing list of chunks; 
            move the first charcter of the current chunk to the first spot 
                in the new chunk
            increment the size of the new chunk;
            move all other characters before the iterator one spot above
            put char c to the spot one above the current spot;
            increment the size of the string;
            return --original iterator;
        } else if (the previous chunk is not full) {
            move the first charcter of the current chunk to the last 
                available spot in the previous chunk
            increment the size of the previous chunk;
            move all other characters before the iterator one spot above
            put char c to the spot one above the current spot;
            increment the size of the string;
            return --original iterator;
        } else, i.e. both the current and previous chunk are full {
            create a new chunk on the heap;
            add the address before the current chunk, i.e. between the current  
                and previous chunk;
            move the first charcter of the current chunk to the first spot 
                in the new chunk
            increment the size of the new chunk;
            move all other characters before the iterator one spot above
            put char c to the spot one above the current spot;
            increment the size of the string;
            return --original iterator;
        }
    }
```


#### `erase(iterator i)`

```
    throw error if string is empty;
    throw error if iterator == end();

    if (iterator points to the last character of a chunk) {
        set the character as null char/ empty char;
        decrement the size of the chunk;
        decrement the size of the string;
        return ++original iterator, i.e. iterator pointing to the first 
            character of the next chunk
    }  else {
        move all characters below the iterator one spot above;
        set the last spot as null char/ empty char. which is a duplicate
        decrement the size of the chunk;
        decrement the size of the string;
        return original iterator;
```

