import torch

def find_intersection(line1, line2, lr=0.01, epochs=1000):
    # Параметры уравнений прямых (ax + by + c = 0)
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    
    # Инициализация случайных координат
    x = torch.tensor([0.0], requires_grad=True)
    y = torch.tensor([0.0], requires_grad=True)

    
    optimizer = torch.optim.SGD([x, y], lr=lr)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        
        # Ошибки для двух уравнений прямых
        loss1 = (a1 * x + b1 * y + c1) ** 2
        loss2 = (a2 * x + b2 * y + c2) ** 2
        
        # Совокупная ошибка
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
    
    return x.item(), y.item()

# Заданные прямые: 3x + 2y - 5 = 0 и 4x - y + 1 = 0
print(f"Прямая 3x + 2y - 5 = 0")
print(f"Прямая 4x - y + 1 = 0")
line1 = (3, 2, -5)
line2 = (4, -1, 1)

intersection = find_intersection(line1, line2)
print("Пересечение в точке:", intersection)
