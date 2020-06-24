import os
import sys
import json
import requests
import xml.etree.ElementTree as ET
import tarfile
import zipfile
import email
from pkg_resources import parse_version

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
        full_name = a.text
        for ext in ['.tar.gz', '.zip', '.whl']:
            if full_name.lower().endswith(ext):
                name, version = full_name.lower().rstrip(ext).split('-')[:2]
                yield (full_name, name, version, link)
                break
        else:
            # print('skipped', full_name)
            continue


def _get_license_and_requirements(url, cache=dict()):
    if url in cache:
        return cache[url]

    dist_name = url.split('/')[-1].split('#')[0]
    dist_path = os.path.join(PACKAGES_DIR, dist_name)

    if not os.path.isfile(dist_path):
        res = requests.get(url)
        res.raise_for_status()
        with open(dist_path, 'wb') as f:
            f.write(res.content)

    if dist_name.endswith('.tar.gz'):
        with tarfile.open(dist_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.endswith('PKG-INFO'):
                    parser = email.parser.HeaderParser()
                    pkg_info = tar.extractfile(member).read().decode()
                    license_name = parser.parsestr(pkg_info)['License']
                    break
            else:
                raise Exception('failed to find PKG-INFO in {}'.format(dist_name))

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
                            requirements.append(l)
                    break
            else:
                print('Warning: failed to find requires.txt in {}'.format(dist_name))

    elif dist_name.endswith('.zip') or dist_name.endswith('.whl'):
        with zipfile.ZipFile(dist_path) as zip_:
            for member in zip_.namelist():
                if member.endswith('metadata.json'):
                    metadata_str = zip_.read(member).decode()
                    metadata = json.loads(metadata_str)
                    if 'license' in metadata:
                        license_name = metadata['license']
                    elif 'License' in metadata:
                        license_name = metadata['License']
                    else:
                        license_name = 'Unknown (failed to find in metadata)'
                    break
            else:
                license_name = 'Unknown (failed to find metadata)'
                print('Warning: failed to find metadata.json in {}'.format(dist_name))
            
            requirements = []
    else:
        raise Exception('exptected tar/zip/whl')
    
    cache[url] = (license_name, requirements)

    return license_name, requirements


def get_package_distributions(package_name, package_version, sign, D, cache=dict(), visited=set()):
    try:
        # sort so that .tag.gz distributions are prioritized
        releases = sorted(
            _iter_pypi(package_name),
            key=lambda x: x[0].endswith('.tar.gz'),
            reverse=True
        )

        # make releases list unqiue by version
        tmp = set()
        new_releases = []
        for full_name, name, version, link in releases:
            if version not in tmp:
                tmp.add(version)
                new_releases.append((full_name, name, version, link))
        releases = new_releases

    except Exception as e:
        print('Warning:', e)
        return

    selected_releases = []

    # if version is not specified, use all releases
    if package_version is None:
        selected_releases.extend(releases)
    
    # otherwise filter them
    else:
        assert sign in ['>=', '<=', '==', '<', '>'], 'invalid sign'
        for full_name, name, version, link in releases:
            if eval('parse_version("{}") {} parse_version("{}")'.format(version, sign, package_version)):
                selected_releases.append((full_name, name, version, link))

    # extract license and requirements from selected releases
    for full_name, name, version, link in selected_releases:
        key = str((name, version))

        if key not in cache:
            try:
                license_name, requirements = _get_license_and_requirements(link)
            except Exception as e:
                cache[key] = str(e)
            else:
                cache[key] = {'license': license_name}
                if key in visited:
                    cache[key]['requirements'] = 'cirular dependency'
                else:
                    cache[key]['requirements'] = dict()
                    for req in requirements:
                        req_name, req_version, req_sign = _split_package_name(req)
                        get_package_distributions(req_name, req_version, req_sign, cache[key]['requirements'], cache, set([*visited, key]))
            print(full_name)
        else:
            print('[using cached]', full_name)

        D[key] = cache[key]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('Usage: python ./{} package_name'.format(__file__))

    name, version, sign = _split_package_name(sys.argv[1])

    D = dict()
    get_package_distributions(name, version, sign, D)
    with open('out.json', 'w') as f:
        f.write(json.dumps(D))