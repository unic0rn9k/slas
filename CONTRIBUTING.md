# Project structure
Rules and notes for how to structure the project

# Undecided
Should matrix deref to tensor-2d or the other way around?

## Lazy code execution

Lazy code execution should be handled by the users of slas.

### Exceptions

There is a more performent way of performing an operation on a lazily computet type.

### Example of when slas should NOT handle lazy code execution
Normalization. There is no operation defined in slas that benefits from knowing that a type has been normalized.
This would mean that all special cases that justify lazy normalization would have to be implemented by the user anyway.

### Example of when slas should handle lazy code execution
Matrix transpose.
Matrix multiplication as defined in slas can be done faster if it is known that a matrix has been lazily transposed.
This means that operations can be performed faster without the user needing to redefine the matrix multiplication operation for lazily transposed matricies.

### How should lazy code execution be handled by the user, when it is not handled by slas?
**I don't know yet, but some possible ideas are:**
 - wrapper types + flagstack + deref to slas types.
