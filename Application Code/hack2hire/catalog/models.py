from django.db import models


class laptop(models.Model):

    product = models.CharField(max_length=100)
    cpu = models.CharField(max_length=100)
    ram = models.PositiveSmallIntegerField()
    price = models.PositiveIntegerField()
    typename = models.CharField(max_length=100)
    inches = models.DecimalField(max_digits=3, decimal_places=1)
    screenresolution = models.CharField(max_length=100)
    memory = models.CharField(max_length=100)
    gpu = models.CharField(max_length=100)
    opsys = models.CharField(max_length=100)
    weight = models.CharField(max_length=20)
    image = models.ImageField(blank=True, null=True)
    recommended = models.PositiveIntegerField(blank=True, null=True, default=0)
    selected = models.PositiveIntegerField(blank=True, null=True, default=0)
    mouse = models.CharField(max_length=100, blank=True, null=True)
    keyboard = models.CharField(max_length=100, blank=True, null=True)
    headset = models.CharField(max_length=100, blank=True, null=True)
    checkout = models.PositiveIntegerField(default=0)

    def __str__(self):
        return self.product

    def numFormat(self):
        # print(type(self.price))
        va = str(self.price)
        if len(va) == 6:
            return va[0:1] + "," + va[1:3] + "," + va[3:]
        elif len(va) == 5:
            return va[0:2] + "," + va[2:]
        elif len(va) == 5:
            return va[0:1] + "," + va[1:]


class accessory(models.Model):

    product = models.CharField(max_length=100)
    protype = models.CharField(max_length=100)
    connectivity = models.CharField(max_length=100)
    price = models.PositiveIntegerField()
    prorange = models.PositiveIntegerField()
    image = models.ImageField(blank=True, null=True)
    selected = models.PositiveIntegerField(blank=True, null=True)

    def __str__(self):
        return self.product

    def numFormat(self):
        # print(type(self.price))
        va = str(self.price)
        if len(va) == 6:
            return va[0:1] + "," + va[1:3] + "," + va[3:]
        elif len(va) == 5:
            return va[0:2] + "," + va[2:]
        elif len(va) == 5:
            return va[0:1] + "," + va[1:]


class cart(models.Model):
    lastAdded = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return self.lastAdded


class useraction(models.Model):
    visitedCount = models.PositiveIntegerField(default=0)
    checkedoutCount = models.PositiveIntegerField(default=0)
