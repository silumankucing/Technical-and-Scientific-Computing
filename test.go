package main

import (
	"fmt"
)

func main() {
	var name string

	// Prompt the user to enter their name
	fmt.Print("Enter your name: ")

	// Read user input from standard input
	fmt.Scanln(&name)

	// Output a greeting message
	fmt.Printf("Hello, %s!\n", name)
}
