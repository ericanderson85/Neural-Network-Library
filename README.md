To use, clone this repository.
```
git clone https://github.com/ericanderson85/mlp/
```

Then, build the project to a jar. 
```
javac -d bin src/main/java/*
jar cvf mlp.jar  -C bin/ .
```

Import the jar into IntelliJ IDEA through project structure.


Example usage of this project with the MNIST handwriting digits dataset:
https://github.com/ericanderson85/DigitRecognizer
