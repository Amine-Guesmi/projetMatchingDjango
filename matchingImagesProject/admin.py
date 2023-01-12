from django.contrib import admin
from .models import Image

class ImagAdmin(admin.ModelAdmin):
    readonly_fields = ('id',)

admin.site.register(Image, ImagAdmin)