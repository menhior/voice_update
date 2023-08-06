from django.db import models

# Create your models here.

class AudioFile(models.Model):
    file = models.FileField(upload_to='audio_files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file.name

class TextFile(models.Model):
    name = models.CharField(max_length=30, blank=True, null=True)
    text = models.CharField(max_length=5000, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)