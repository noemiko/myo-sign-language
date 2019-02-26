from setuptools import find_packages
from setuptools import setup


def readme():
    with open('../README.md') as f:
        return f.read()


if __name__ == '__main__':
    setup(
        name='myo_sign_language',
        long_description=readme(),
        version='0.1',
        url='https://github.com/noemiko/myo-sign-language',
        author_email='noemiko8@gmail.com',
        author='noemiko',
        packages=find_packages(),
        license='MIT',
        zip_safe=False
    )
