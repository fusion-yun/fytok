__path__ = __import__("pkgutil").extend_path(__path__, __name__)


# from importlib import resources
# from fytok.utils.logger import logger

# mapping_path = [item.resolve() for item in resources.files("fytok.mapping") if item.is_dir()]

# logger.verbose(f"Mapping path: {mapping_path}")

# import spdm.core.mapper as mapper

# mapper.path.extend(mapping_path)

# from importlib import resources as impresources
# try:
#     import spdm.core.entry as entry_
#     entry_._mapping_path.extend([p.resolve() for p in impresources.files(mapping)._paths])
# except Exception as error:
#     raise FileNotFoundError(f"Can not find mappings!") from error
