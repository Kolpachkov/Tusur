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
            raise OverflowError("Очередь полная")
        self.queue[self.rear] = value
        self.rear = (self.rear + 1) % self.size
        self.count += 1

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Очередь пустая")
        value = self.queue[self.front]
        self.front = (self.front + 1) % self.size
        self.count -= 1
        return value

    def is_empty(self):
        return self.count == 0

    def display(self):
        if self.is_empty():
            print("Очередь пустая")
        else:
            print("Очередь:", end=" ")
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
            raise IndexError("Очередь пустая")
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
            print("Очередь пустая")
        else:
            print("Очередь:", end=" ")
            for i in range(self.count):
                print(self.queue[(self.front + i) % self.size], end=" ")
            print()

def user_select_queue():
    print("Выберите тип очереди:")
    print("1. Статическая очередь")
    print("2. Динамическая очередь")
    choice = input("Введите номер вашего выбора: ")
    
    size = int(input("Введите размер очереди: "))
    
    match choice:
        case "1":
            return StaticArrayQueue(size)
        case "2":
            return DynamicArrayQueue(size)
        case _:
            print("Неверный выбор, создается очередь по умолчанию (динамическая).")
            return DynamicArrayQueue(size)


if __name__ == "__main__":
    queue = user_select_queue()

    while True:
        print("\n1. Добавить элемент в очередь")
        print("2. Удалить элемент из очереди")
        print("3. Показать очередь")
        print("4. Выход")
        action = input("Выберите действие: ")
        
        match action:
            case "1":
                value = int(input("Введите элемент для добавления: "))
                try:
                    queue.enqueue(value)
                except OverflowError as e:
                    print(e)
            case "2":
                try:
                    print("Удалён элемент:", queue.dequeue())
                except IndexError as e:
                    print(e)
            case "3":
                queue.display()
            case "4":
                break
            case _:
                print("Неверный выбор, попробуйте снова.")
