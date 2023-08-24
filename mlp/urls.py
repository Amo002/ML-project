
from django.contrib import admin
from django.urls import path, include, re_path
from django.views.generic.base import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('mlproject.urls')),
    re_path(r'^$', RedirectView.as_view(url='results/', permanent=False)),
]

