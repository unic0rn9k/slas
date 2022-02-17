//use std::any::TypeId;

// Maybe replace type tag enum with bool.
// This whay a type tag will literally just be a compiletime stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeTag {
    IsNormalized,
    IsMatrix,
    IsTransposed,
    IsTagList,
    Revoked,
    Max,
}

pub use TypeTag::*;

impl const From<usize> for TypeTag {
    fn from(u: usize) -> Self {
        if u > Max as usize {
            panic!("Tried to convert usize bigger than TypeTag::Max to TypeTag")
        } else {
            unsafe { std::mem::transmute(u as u8) }
        }
    }
}

pub trait DynTaggable {
    fn has_tag(&self, tag: TypeTag) -> bool;

    fn add_tag(&mut self, tag: TypeTag);
    fn del_tag(&mut self, tag: TypeTag);
}

pub trait ConstTaggable: DynTaggable {
    fn const_has_tag<const TAG: usize>(&self) -> bool
    where
        Self: ~const ConstTaggable,
        Self: Sized;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DynTypeTag(TypeTag);

impl DynTaggable for DynTypeTag {
    fn has_tag(&self, tag: TypeTag) -> bool {
        tag == self.0
    }

    fn add_tag(&mut self, tag: TypeTag) {
        self.0 = tag
    }

    fn del_tag(&mut self, tag: TypeTag) {
        if self.0 == tag {
            self.0 = Revoked
        } else {
            panic!(
                "Cannot revoke TypeTag::{:?} that doesn't match tag found in self ({:?})",
                tag, self.0
            )
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstTypeTag<const TAG: usize>;

impl<const TAG: usize> const ConstTaggable for ConstTypeTag<TAG> {
    fn const_has_tag<const TAG2: usize>(&self) -> bool {
        TAG == TAG2
    }
}

impl<const TAG: usize> const DynTaggable for ConstTypeTag<TAG> {
    fn has_tag(&self, tag: TypeTag) -> bool {
        tag as usize == TAG
    }

    fn add_tag(&mut self, _: TypeTag) {
        panic!("Cannot apply TypeTag to ConstTypeTag")
    }
    fn del_tag(&mut self, _: TypeTag) {
        panic!("Cannot revoke TypeTag from ConstTypeTag")
    }
}

pub struct TypeTagStack<'a, const TAG: usize>(&'a dyn ConstTaggable);

impl<'a, const TAG: usize> DynTaggable for TypeTagStack<'a, TAG> {
    fn has_tag(&self, tag: TypeTag) -> bool {
        if TAG == IsTagList as usize || tag as usize == TAG {
            true
        } else {
            self.0.has_tag(tag)
        }
    }

    fn add_tag(&mut self, _tag: TypeTag) {
        panic!("Please use LinkedTypeTagList::push_tag instead")
    }
    fn del_tag(&mut self, _tag: TypeTag) {
        panic!("Please use LinkedTypeTagList::pop_tag instead")
    }
}

impl<'a, const TAG: usize> ConstTaggable for TypeTagStack<'a, TAG> {
    fn const_has_tag<const TAG2: usize>(&self) -> bool {
        if TAG == IsTagList as usize || TAG2 == TAG {
            true
        } else {
            self.0.has_tag(TAG2.into())
        }
    }
}

impl<'a, const TAG: usize> TypeTagStack<'a, TAG> {
    pub const fn push_tag<const TAG2: usize>(&'a self) -> TypeTagStack<'a, TAG2> {
        TypeTagStack(self)
    }
    pub const fn pop_tag(&'a self) -> &'a dyn ConstTaggable {
        self.0
    }
}
