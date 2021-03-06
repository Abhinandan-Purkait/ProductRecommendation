# Generated by Django 3.0 on 2019-12-11 21:59

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='accessory',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('product', models.CharField(max_length=100)),
                ('protype', models.CharField(max_length=100)),
                ('connectivity', models.CharField(max_length=100)),
                ('price', models.PositiveIntegerField()),
                ('prorange', models.PositiveIntegerField()),
                ('image', models.ImageField(default='/home/saltyclown/Desktop/Dell/hack2hire/static/img/grey-box.png', upload_to='')),
                ('selected', models.PositiveIntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='laptop',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('product', models.CharField(max_length=100)),
                ('cpu', models.CharField(max_length=100)),
                ('ram', models.PositiveSmallIntegerField()),
                ('price', models.PositiveIntegerField()),
                ('typename', models.CharField(max_length=100)),
                ('inches', models.DecimalField(decimal_places=1, max_digits=3)),
                ('screenresolution', models.CharField(max_length=100)),
                ('memory', models.CharField(max_length=100)),
                ('gpu', models.CharField(max_length=100)),
                ('opsys', models.CharField(max_length=100)),
                ('weight', models.CharField(max_length=20)),
                ('image', models.ImageField(default='/home/saltyclown/Desktop/Dell/hack2hire/static/img/grey-box.png', upload_to='')),
                ('recommended', models.PositiveIntegerField()),
                ('selected', models.PositiveIntegerField()),
            ],
        ),
    ]
