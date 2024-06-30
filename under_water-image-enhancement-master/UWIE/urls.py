from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .import views

urlpatterns = [
	path('', views.index, name='index'),
    path('home', views.index, name='index'),
    path('clahe',views.clahe,name='clahe'),
    path('enhance',views.get_image,name='getimage'),
    path('rayleigh',views.rayleigh,name='rayleigh'),
    path('mip',views.mip,name='mip'),
    path('dcp',views.dcp,name='dcp'),
    path('enhanceray',views.get_image_ray,name='getimageray'),
    path('restoremip',views.get_image_mip,name='getimagemip'),
    path('restoredcp',views.get_image_mip,name='getimagedcp'),
    path('both',views.both,name='both'),
    path('enhanceboth',views.get_image_both,name='getimageboth'),
    path('auto',views.auto,name='auto'),
    path('autogetimage',views.autogetimage,name='autogetimage'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)