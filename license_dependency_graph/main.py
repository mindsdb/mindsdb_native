import os
import sys
import requests
import xml.etree.ElementTree as ET
from collections import defaultdict
import tarfile
import email
from packaging.version import parse as parse_version

PACKAGES_DIR = './packages'
if not os.path.isdir(PACKAGES_DIR):
    os.makedirs(PACKAGES_DIR)


def _split_package_name(package_name):
    """
    converts str of format "name>=x.y.z" to tuple(name, str('x.y.z'))
    """

    for sep in ['>=', '<=', '==', '<', '>']:
        tmp = package_name.split(sep)
        if len(tmp) == 2:
            name, version, sign = tmp[0].strip(), tmp[1].strip(), sep
            break
    else:
        name, version, sign = package_name.strip(), None, None

    return name, version, sign


def _iter_pypi(package_name):
    res = requests.get('https://pypi.org/simple/{}'.format(package_name))
    res.raise_for_status()
    tree = ET.fromstring(res.text)
    for a in tree.findall('.//a'):
        link = a.get('href')
        name = a.text.lower()
        if not name.endswith('.tar.gz'):
            print('skipped', name)
            continue
        name, version = name.rstrip('.tar.gz').split('-')[:2]
        yield (a.text, name, version, link)


def _get_license_and_requirements(tar_url):
    tar_name = tar_url.split('/')[-1].split('#')[0]
    tar_path = os.path.join(PACKAGES_DIR, tar_name)

    if not os.path.isfile(tar_path):
        res = requests.get(tar_url)
        res.raise_for_status()
        with open(tar_path, 'wb') as f:
            f.write(res.content)

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith('PKG-INFO'):
                parser = email.parser.HeaderParser()
                pkg_info = tar.extractfile(member).read().decode()
                license_name = parser.parsestr(pkg_info)['License']
                break
        else:
            raise Exception('failed to find PKG-INFO in {}'.format(tar_name))
        
        requirements = []
        for member in tar.getmembers():
            if member.name.endswith('requires.txt'):
                lines = tar.extractfile(member).read().decode().split('\n')
                requirements = []
                for l in lines:
                    # this is temporary
                    if l.startswith('['):
                        break
                    else:
                        print('req1', l)
                        requirements.append(l)
                break
        else:
            print('Warning: failed to find requires.txt in {}'.format(tar_name))

        return license_name, requirements

def get_package_distributions(package_name, package_version, sign, d):
    releases = list(_iter_pypi(package_name))

    selected_releases = []

    if package_version is None:
        selected_releases.extend(releases)
    else:
        assert sign in ['>=', '<=', '==', '<', '>'], 'invalid sign'
        for full_name, name, version, link in releases:
            if eval('"{}" {} "{}"'.format(parse_version(version), sign, parse_version(package_version))):
                selected_releases.append((full_name, name, version, link))
                break
        else:
            raise Exception('version {} not found'.format(package_version))
            
    for full_name, name, version, link in selected_releases:
        license_name, requirements = _get_license_and_requirements(link)
        d[full_name].add(license_name)
        for req in requirements:
            req_name, req_version, req_sign = _split_package_name(req)
            print('req2', req_name, req_version, req_sign)
            get_package_distributions(req_name, req_version, req_sign, d)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('Usage: python ./{} package_name'.format(__file__))

    name, version, sign = _split_package_name(sys.argv[1])
    d = defaultdict(set)
    get_package_distributions(name, version, sign, d)
    for k, v in d.items():
        print(k, v)
        