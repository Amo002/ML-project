from django.urls import path
from . import views

urlpatterns = [
    path('results/', views.select_model, name='select_model'),
    path('results/<str:model_name>/', views.display_results, name='display_results'),
]

