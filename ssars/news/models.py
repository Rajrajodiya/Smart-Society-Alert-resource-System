from django.db import models

# Create your models here.
class News(models.Model):
    headline=models.CharField(max_length=200)
    body=models.TextField()
    date=models.DateField()
    imageURL=models.TextField(null=True, blank=True)
    def __str__(self):
        return self.headline