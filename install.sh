rm -rf build
rm -rf dist
rm -rf mindsdb.egg-info

echo "mode (prod/dev)?"

read mode

if [ "$mode" = "prod" ]; then

    python3 setup.py develop --uninstall
    python3 setup.py clean
    python3 setup.py build
    python3 setup.py sdist

fi

if [ "$mode" = "dev" ]; then
    pip3 uninstall mindsdb-native
    pip3 uninstall mindsdb-native --user
    python3 setup.py develop --uninstall
    python3 setup.py develop
fi
