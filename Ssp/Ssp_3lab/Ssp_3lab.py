from abc import ABC, abstractmethod

class Queue(ABC):
    @abstractmethod
    def enqueue(self, value):
        pass

    @abstractmethod
    def dequeue(self):
        pass

    @abstractmethod
    def is_empty(self):
        pass

    @abstractmethod
    def display(self):  
        pass

class StaticArrayQueue(Queue):
    def __init__(self, size=5):
        self.size = size
        self.queue = [None] * size
        self.front = 0
        self.rear = 0
        self.count = 0

    def enqueue(self, value):
        if self.count == self.size:
            raise OverflowError("Queue is full")
        self.queue[self.rear] = value
        self.rear = (self.rear + 1) % self.size
        self.count += 1

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        value = self.queue[self.front]
        self.front = (self.front + 1) % self.size
        self.count -= 1
        return value

    def is_empty(self):
        return self.count == 0

    def display(self):
        if self.is_empty():
            print("Queue is empty")
        else:
            print("Queue:", end=" ")
            for i in range(self.count):
                print(self.queue[(self.front + i) % self.size], end=" ")
            print()

class DynamicArrayQueue(Queue):
    def __init__(self, size=5):
        self.queue = [None] * size
        self.front = 0
        self.rear = 0
        self.size = size
        self.count = 0

    def enqueue(self, value):
        if self.count == self.size:
            self._resize(self.size * 2)  
        self.queue[self.rear] = value
        self.rear = (self.rear + 1) % self.size
        self.count += 1

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        value = self.queue[self.front]
        self.front = (self.front + 1) % self.size
        self.count -= 1
        return value

    def is_empty(self):
        return self.count == 0

    def _resize(self, new_size):
        new_queue = [None] * new_size
        for i in range(self.count):
            new_queue[i] = self.queue[(self.front + i) % self.size]
        self.queue = new_queue
        self.front = 0
        self.rear = self.count
        self.size = new_size

    def display(self):
        if self.is_empty():
            print("Queue is empty")
        else:
            print("Queue:", end=" ")
            for i in range(self.count):
                print(self.queue[(self.front + i) % self.size], end=" ")
            print()


if __name__ == "__main__":
    static_queue = StaticArrayQueue(3)
    dynamic_queue = DynamicArrayQueue(3)

    print("StaticArrayQueue:")
    static_queue.enqueue(1)
    static_queue.enqueue(2)
    static_queue.display()  
    print(static_queue.dequeue())
    static_queue.display()  

    print("\nDynamicArrayQueue:")
    dynamic_queue.enqueue(10)
    dynamic_queue.enqueue(20)
    dynamic_queue.enqueue(30)
    dynamic_queue.enqueue(40)  
    dynamic_queue.display()  
    print(dynamic_queue.dequeue())
    dynamic_queue.display()  
