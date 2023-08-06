from django.contrib import admin

# Register your models here.
# Register your models here.
from .models import AudioFile, TextFile

admin.site.register(TextFile)
admin.site.register(AudioFile)