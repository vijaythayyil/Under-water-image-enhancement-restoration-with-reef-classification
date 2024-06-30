from django.db import models
# Create your models here.


class Input(models.Model):
	img = models.ImageField(upload_to = "UWIE/static/Input/")
	
	def __str__(self):
		return self.img