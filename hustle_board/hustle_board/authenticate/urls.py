from django.urls import path
from . import views


urlpatterns = [
    path('', views.login_user, name="home"),
    path('login/', views.home, name="login"),
    path('logout/', views.logout_user, name="logout"),
    path('profile/', views.profile, name="profile"),
    path('register/', views.register_user, name="register"),
    path('test/', views.test, name="test"),




]
